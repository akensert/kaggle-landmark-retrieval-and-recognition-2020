import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection
import tqdm
import os

from generator import create_dataset
from models import Delf
from optimizer import get_optimizer
from config import config_1 as config

# import logging
# tf.get_logger().setLevel(logging.ERROR)
# import warnings
# warnings.filterwarnings("ignore")

class DistributedModel:

    def __init__(self, backbone, input_size, n_classes, phases,
                 batch_size, dense_units, dropout_rate, gem_p, loss,
                 scale, margin, clip_grad, checkpoint_weights,
                 optimizer, strategy, mixed_precision):

        self.model = Delf(n_classes, scale, margin, name='DELF')

        self.input_sizes = input_size
        self.batch_sizes = batch_size
        self.phases = phases

        if checkpoint_weights:
            self.delf_model.load_weights(checkpoint_weights + '.h5')

        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.strategy = strategy
        self.mixed_precision = mixed_precision

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.train_desc_loss = tf.keras.metrics.Mean()
        self.train_attn_loss = tf.keras.metrics.Mean()
        self.train_desc_accu = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
        self.train_attn_accu = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, logits):
        per_example_loss = self.loss_object(labels, logits)
        return tf.nn.compute_average_loss(
            per_example_loss,
            global_batch_size=(
                self.batch_size * self.strategy.num_replicas_in_sync)
            )

    def _backprop_loss(self, tape, loss, weights):
        gradients = tape.gradient(loss, weights)
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_grad)
        self.optimizer.apply_gradients(zip(clipped, weights))


    def _train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as desc_tape:
            probs, feat = self.model.forward_prop_desc(images, labels, training=True)
            desc_loss = self._compute_loss(labels, probs)
            self.train_desc_loss.update_state(desc_loss)
            self.train_desc_accu.update_state(labels, probs)
            if self.mixed_precision:
                desc_loss = self.optimizer.get_scaled_loss(desc_loss)

        self._backprop_loss(desc_tape, desc_loss, self.model.get_descriptor_weights)

        with tf.GradientTape() as attn_tape:
            feat = tf.stop_gradient(feat)
            probs = self.model.forward_prop_attn(feat, training=True)
            attn_loss = self._compute_loss(labels, probs)
            self.train_attn_loss.update_state(attn_loss)
            self.train_attn_accu.update_state(labels, probs)
            if self.mixed_precision:
                attn_loss = self.optimizer.get_scaled_loss(attn_loss)

        self._backprop_loss(attn_tape, attn_loss, self.model.get_attention_weights)

        return desc_loss, attn_loss

    @tf.function(experimental_relax_shapes=True)
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss = self.strategy.run(
            self._train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_loss,
            axis=None
        )

    def train(self, train_df, epochs, sample_frac, save_path):

        for epoch in range(epochs):

            phase = 0
            for p in self.phases:
                if epoch >= p:
                    phase += 1

            self.batch_size = self.batch_sizes[phase]
            self.input_size = self.input_sizes[phase]

            print(f"Phase {phase}, input_size={self.input_size}, batch_size={self.batch_size}")

            train_ds = create_dataset(
                    dataframe=train_df,
                    training=True,
                    sample_frac=sample_frac,
                    batch_size=self.batch_size,
                    input_size=self.input_size,
                )

            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            train_ds = tqdm.tqdm(train_ds)
            for i, inputs in enumerate(train_ds):
                _, _ = self._distributed_train_step(inputs)
                train_ds.set_description(
                    "Loss {:.3f} {:.3f}, Acc {:.3f} {:.3f}".format(
                        self.train_desc_loss.result().numpy(),
                        self.train_attn_loss.result().numpy(),
                        self.train_desc_accu.result().numpy(),
                        self.train_attn_accu.result().numpy(),
                    )
                )

            if save_path:
                self.model.save_weights(save_path + f'_{phase}' + '.h5')

            self.train_desc_loss.reset_states()
            self.train_attn_loss.reset_states()
            self.train_desc_accu.reset_states()
            self.train_attn_accu.reset_states()


gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
mixed_precision = False
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    mixed_precision = True

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


def prepare_dataframe(df_orig, alpha=1.0):
    df = df_orig.copy()
    repl_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['weight'] = 1 / df['landmark_id'].map(repl_map).astype(np.float32)**alpha
    df['label'] = df['label'].astype(np.int32)
    df['image_target_ratio'] = df['image_target_ratio'].astype(np.float32)
    return df

dataframe = pd.read_csv('../input/modified_train.csv')
#dataframe = dataframe.iloc[::150]
dataframe = prepare_dataframe(dataframe, alpha=config['data_sampling']['alpha'])


with strategy.scope():

    optimizer = get_optimizer(
        opt=config['optimizer'],
        steps_per_epoch=config['learning_rate']['steps_per_epoch'],
        lr_max=config['learning_rate']['max'],
        lr_min=config['learning_rate']['min'],
        warmup_epochs=config['learning_rate']['warmup_epochs'],
        decay_epochs=config['learning_rate']['decay_epochs'],
        power=config['learning_rate']['power'],
    )

    dist_model = DistributedModel(
        backbone=config['backbone'],
        input_size=config['input_size'],
        n_classes=config['n_classes'],
        phases=config['phases'],
        batch_size=config['batch_size'],
        dense_units=config['dense_units'],
        dropout_rate=config['dropout_rate'],
        gem_p=config['gem_p'],
        loss=config['loss']['type'],
        scale=config['loss']['scale'],
        margin=config['loss']['margin'],
        clip_grad=config['clip_grad'],
        checkpoint_weights=config['checkpoint_weights'],
        optimizer=optimizer,
        strategy=strategy,
        mixed_precision=mixed_precision)

    dist_model.train(
        train_df=dataframe,
        epochs=config['n_epochs'],
        sample_frac=config['data_sampling']['frac'],
        save_path=config['save_path'])
