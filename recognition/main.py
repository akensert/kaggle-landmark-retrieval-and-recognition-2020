"""
What's hardcoded:

    generator.py -> normalize()
    model.py     -> Delf() -> _architectures{}
    serve.py     -> ServedModel() -> extraction.compute_receptive_boxes()
    serve.py & main.py -> load_weights() & save_weights() respectively

    FIX:
        serve.py label input to prediction functions
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
                 learning_rate_start,
                 learning_rate_end,
                 learning_momentum,
                 dense_units,
                 margin_type,
                 scale,
                 margin,
                 checkpoint_weights,
                 strategy,
                 mixed_precision):

        self.model = Delf(
            dense_units=dense_units,
            margin_type=margin_type,
            scale=scale,
            margin=margin,
            input_dim=input_dim,
            name='DELF')

        self.input_dim = input_dim
        self.batch_size = batch_size

        if checkpoint_weights:
            self.model.build([[None, input_dim, input_dim, 3], [None]])
            self.model.load_weights(
                '../output/weights/' + self.model.backbone.name + '.h5')


        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_start,
            momentum=learning_momentum,
        )

        self.strategy = strategy
        self.mixed_precision = mixed_precision

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.train_desc_loss = tf.keras.metrics.Mean()
        self.train_attn_loss = tf.keras.metrics.Mean()
        self.train_desc_accu = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
        self.train_attn_accu = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    self.optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, logits, sample_weights):
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
        self.optimizer.apply_gradients(zip(gradients, weights))

    # def _train_step(self, inputs):
    #
    #     images, labels = inputs
    #
    #     with tf.GradientTape() as desc_tape, tf.GradientTape() as attn_tape:
    #
    #         desc_probs, intermediate_feat = self.model.forward_prop_desc(
    #             images, labels, training=True)
    #         # intermediate_feat = tf.stop_gradient(intermediate_feat)
    #         attn_probs = self.model.forward_prop_attn(
    #             intermediate_feat, training=True)
    #
    #         desc_loss = self._compute_loss(labels, desc_probs)
    #         attn_loss = self._compute_loss(labels, attn_probs)
    #
    #         self.train_desc_loss.update_state(desc_loss)
    #         self.train_desc_accu.update_state(labels, desc_probs)
    #         self.train_attn_loss.update_state(attn_loss)
    #         self.train_attn_accu.update_state(labels, attn_probs)
    #
    #         if self.mixed_precision:
    #             desc_loss = self.optimizer.get_scaled_loss(desc_loss)
    #             attn_loss = self.optimizer.get_scaled_loss(attn_loss)
    #
    #     self._backprop_loss(desc_tape, desc_loss, self.model.get_descriptor_weights)
    #     self._backprop_loss(attn_tape, attn_loss, self.model.get_attention_weights)
    #
    #     return desc_loss, attn_loss


    def _train_step(self, inputs):

        images, labels, sample_weights = inputs

        with tf.GradientTape() as desc_tape:
            desc_probs, intermediate_feat = self.model.forward_prop_desc(
                images, labels, training=True)
            desc_loss = self._compute_loss(labels, desc_probs, sample_weights)
            self.train_desc_loss.update_state(desc_loss)
            self.train_desc_accu.update_state(labels, desc_probs)

            if self.mixed_precision:
                desc_loss = self.optimizer.get_scaled_loss(desc_loss)

        self._backprop_loss(desc_tape, desc_loss, self.model.descriptor_weights)

        with tf.GradientTape() as attn_tape:
            intermediate_feat = tf.stop_gradient(intermediate_feat)
            attn_probs = self.model.forward_prop_attn(
                intermediate_feat, labels, training=True)
            attn_loss = self._compute_loss(labels, attn_probs, sample_weights)
            self.train_attn_loss.update_state(attn_loss)
            self.train_attn_accu.update_state(labels, attn_probs)

            if self.mixed_precision:
                attn_loss = self.optimizer.get_scaled_loss(attn_loss)

        self._backprop_loss(attn_tape, attn_loss, self.model.attention_weights)

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

    @staticmethod
    def cosine_decay(step, lr_start, lr_end, total_steps):
        alpha = lr_end / lr_start
        decay = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        decay = (1 - alpha) * decay + alpha
        return lr_start * decay

    def train(self, train_df, epochs):

        train_ds = create_dataset(
            dataframe=train_df,
            training=True,
            batch_size=self.batch_size,
            target_dim=self.input_dim,
            central_crop=False,
            crop_ratio=(0.7, 1.0),
            apply_augmentation=True
        )

        num_iterations = 400_000//self.batch_size#len(train_ds) // 4
        total_num_iterations = float(num_iterations * epochs)

        train_ds = self.strategy.experimental_distribute_dataset(train_ds)

        for epoch in range(epochs):
            pbar = tqdm.tqdm(total=num_iterations)
            for inputs in train_ds:
                _, _ = self._distributed_train_step(inputs)
                pbar.update(n=1)
                pbar.set_description(
                    "LR {:.4f} - Loss {:.3f} {:.3f} - Acc {:.3f} {:.3f}".format(
                        self.optimizer.learning_rate.numpy(),
                        self.train_desc_loss.result().numpy(),
                        self.train_attn_loss.result().numpy(),
                        self.train_desc_accu.result().numpy(),
                        self.train_attn_accu.result().numpy(),
                    )
                )

                self.optimizer.learning_rate = self.cosine_decay(
                    step=self.optimizer.iterations.numpy()/2,
                    lr_start=self.learning_rate_start,
                    lr_end=self.learning_rate_end,
                    total_steps=total_num_iterations,
                )

            pbar.close()

            self.model.save_weights(
                '../output/weights/' + self.model.backbone.name + '.h5')

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
    df['prob'] = ((1/np.sqrt(counts)) / (1/np.sqrt(counts)).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df

# def read_train_file(input_path):
#     files_paths = glob.glob(input_path + 'train/*/*/*/*')
#     mapping = {}
#     for path in files_paths:
#         mapping[path.split('/')[-1].split('.')[0]] = path
#     df = pd.read_csv(input_path + 'train.csv')
#     df['path'] = df['id'].map(mapping)
#     counts_map = dict(
#         df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
#     counts = df['landmark_id'].map(counts_map)
#     uniques = df['landmark_id'].unique()
#     df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
#     df['weight'] = df.label.map(dict(zip(
#             df.label.unique(),
#             np.log(np.bincount(df.label)).sum() / (df.label.nunique() * np.log(np.bincount(df.label)))
#     )))
#     return df

train_df = read_train_file('../input/')

with strategy.scope():

    dist_model = DistributedModel(
        input_dim=config['input_dim'],
        batch_size=config['batch_size'],
        learning_rate_start=config['optimizer']['learning_rate_start'],
        learning_rate_end=config['optimizer']['learning_rate_end'],
        learning_momentum=config['optimizer']['momentum'],
        dense_units=config['dense_units'],
        margin_type=config['loss']['type'],
        scale=config['loss']['scale'],
        margin=config['loss']['margin'],
        checkpoint_weights=config['checkpoint_weights'],
        strategy=strategy,
        mixed_precision=mixed_precision)

    dist_model.train(train_df=train_df, epochs=config['n_epochs'])
