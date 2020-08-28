"""
What's hardcoded:

    generator.py -> normalize()
    model.py     -> Delf() -> _architectures{}
    serve.py     -> ServedModel() -> extraction.compute_receptive_boxes()
"""


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math
import pandas as pd
import tqdm
import os
import glob


from generator import create_dataset
from models import Delf
from config import config

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

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


class DistributedModel:

    def __init__(self,
                 input_dim,
                 batch_size,
                 dense_units,
                 margin_type,
                 scale,
                 margin,
                 checkpoint_weights,
                 optimizer,
                 strategy,
                 mixed_precision):

        self.model = Delf(
            dense_units=dense_units,
            margin_type=margin_type,
            scale=scale,
            margin=margin,
            input_dim=input_dim,
            name='DELF')

        self.clip_grad = 10
        self.input_dim = input_dim
        self.batch_size = batch_size

        if checkpoint_weights:
            self.model.build([[None, input_dim, input_dim, 3], [None]])
            self.model.load_weights(checkpoint_weights + '.h5')

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

        with tf.GradientTape() as desc_tape, tf.GradientTape() as attn_tape:

            desc_probs, intermediate_feat = self.model.forward_prop_desc(
                images, labels, training=True)
            intermediate_feat = tf.stop_gradient(intermediate_feat)
            attn_probs = self.model.forward_prop_attn(
                intermediate_feat, training=True)

            desc_loss = self._compute_loss(labels, desc_probs)
            attn_loss = self._compute_loss(labels, attn_probs)

            self.train_desc_loss.update_state(desc_loss)
            self.train_desc_accu.update_state(labels, desc_probs)
            self.train_attn_loss.update_state(attn_loss)
            self.train_attn_accu.update_state(labels, attn_probs)

            if self.mixed_precision:
                desc_loss = self.optimizer.get_scaled_loss(desc_loss)
                attn_loss = self.optimizer.get_scaled_loss(attn_loss)

        self._backprop_loss(desc_tape, desc_loss, self.model.get_descriptor_weights)
        self._backprop_loss(attn_tape, attn_loss, self.model.get_attention_weights)

        return desc_loss, attn_loss

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_repl_desc_loss, per_repl_attn_loss = self.strategy.run(
            self._train_step, args=(dist_inputs,))
        return (
        self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_repl_desc_loss,
            axis=None),
        self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_repl_attn_loss,
            axis=None)
        )

    def train(self, train_df, epochs, save_path):

        for epoch in range(epochs):

            train_ds = create_dataset(
                dataframe=train_df,
                training=True,
                batch_size=self.batch_size,
                target_dim=self.input_dim,
                central_crop=False,
                crop_ratio=(0.7, 1.0),
                apply_augmentation=True
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
                self.model.save_weights(save_path + '.h5')

            self.train_desc_loss.reset_states()
            self.train_attn_loss.reset_states()
            self.train_desc_accu.reset_states()
            self.train_attn_accu.reset_states()


def read_train_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)
    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    counts = df['landmark_id'].map(counts_map)
    df['prob'] = ((1/counts**alpha) / (1/counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df

train_df = read_train_file('../input/')

with strategy.scope():

    dist_model = DistributedModel(
        input_dim=config['input_dim'],
        batch_size=config['batch_size'],
        dense_units=config['dense_units'],
        margin_type=config['loss']['type'],
        scale=config['loss']['scale'],
        margin=config['loss']['margin'],
        checkpoint_weights=config['checkpoint_weights'],
        optimizer=tfa.optimizers.SGDW(
            weight_decay=config['optimizer']['weight_decay'],
            learning_rate=config['optimizer']['learning_rate'],
            momentum=config['optimizer']['momentum']),
        strategy=strategy,
        mixed_precision=mixed_precision)

    dist_model.train(
        train_df=train_df,
        epochs=config['n_epochs'],
        save_path=config['save_path'])
