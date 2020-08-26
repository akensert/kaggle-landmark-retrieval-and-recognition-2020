import numpy as np
import tensorflow as tf
import math
import pandas as pd
import tqdm
import os
import glob

from generator import create_dataset
from models import Delf
from optimizer import get_optimizer
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

    def __init__(self, backbone, input_dim, n_classes,
                 batch_size, dense_units, dropout_rate, gem_p, loss,
                 scale, margin, clip_grad, checkpoint_weights,
                 optimizer, strategy, mixed_precision):

        self.model = Delf(
            dense_units, n_classes, gem_p, scale, margin,
            input_dim=input_dim, backbone=backbone, name='DELF')

        self.input_dim = input_dim
        self.batch_size = batch_size


        if checkpoint_weights:
            self.model.build([[None, input_dim, input_dim, 3], [None]])
            self.model.load_weights(checkpoint_weights + '.h5')

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

            probs, feat_block4 = self.model.forward_prop_desc(images, labels, training=True)
            desc_loss = self._compute_loss(labels, probs)

            self.train_desc_loss.update_state(desc_loss)
            self.train_desc_accu.update_state(labels, probs)

            if self.mixed_precision:
                desc_loss = self.optimizer.get_scaled_loss(desc_loss)

        self._backprop_loss(desc_tape, desc_loss, self.model.get_descriptor_weights)

        with tf.GradientTape() as attn_tape:

            feat_block4 = tf.stop_gradient(feat_block4)
            probs = self.model.forward_prop_attn(feat_block4, training=True)

            attn_loss = self._compute_loss(labels, probs)

            self.train_attn_loss.update_state(attn_loss)
            self.train_attn_accu.update_state(labels, probs)

            if self.mixed_precision:
                attn_loss = self.optimizer.get_scaled_loss(attn_loss)

        self._backprop_loss(attn_tape, attn_loss, self.model.get_attention_weights)

        return desc_loss, attn_loss

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_repl_loss1, per_repl_loss2 = self.strategy.run(
            self._train_step, args=(dist_inputs,))
        return (
        self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_repl_loss1,
            axis=None),
        self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_repl_loss2,
            axis=None)
        )

    def train(self, train_df, epochs, undersample, save_path):

        for epoch in range(epochs):

            if epoch == 36:
                self.batch_size = 10
                self.input_dim = 512

            train_ds = create_dataset(
                dataframe=train_df,
                undersample=undersample,
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


def read_submission_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'test/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'sample_submission.csv')
    df['path'] = df['id'].map(mapping)
    df['label'] = -1
    df['prob'] = -1
    return df

def read_train_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)

    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['prob'] = (
        (1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df, dict(zip(df.label, df.landmark_id))


submission_df = read_submission_file('../input/')
train_df, mapping = read_train_file('../input/')

print("train shape      =", train_df.shape)
print("submission shape =", submission_df.shape)

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
        input_dim=config['input_dim'],
        n_classes=config['n_classes'],
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
        train_df=train_df,
        epochs=config['n_epochs'],
        undersample=config['undersample'],
        save_path=config['save_path'])
