import tensorflow as tf
import os
import tqdm
import pandas as pd
import numpy as np

from models.efficientnet_sync import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7)
from models.resnet_sync import (
    ResNet50, ResNet101, ResNet152)
from models.densenet_sync import (
    DenseNet121, DenseNet169, DenseNet201)
from models.inception_sync import (
    InceptionV3)
from models.inceptionresnet_sync import (
    InceptionResNetV2)
from models.xception_sync import (
    Xception)

from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct
from generator import create_dataset


CNN_ARCHITECTURES = {
    'efficientnet-b0':    EfficientNetB0,
    'efficientnet-b1':    EfficientNetB1,
    'efficientnet-b2':    EfficientNetB2,
    'efficientnet-b3':    EfficientNetB3,
    'efficientnet-b4':    EfficientNetB4,
    'efficientnet-b5':    EfficientNetB5,
    'efficientnet-b6':    EfficientNetB6,
    'efficientnet-b7':    EfficientNetB7,
    'resnet-50':          ResNet50,
    'resnet-101':         ResNet101,
    'resnet-152':         ResNet152,
    'densenet-121':       DenseNet121,
    'densenet-169':       DenseNet169,
    'densenet-201':       DenseNet201,
    'inception-v3':       InceptionV3,
    'inceptionresnet-v2': InceptionResNetV2,
    'xception':           Xception,
}

def create_model(backbone, input_size, n_classes, gem_p,
                 dense_units, dropout_rate, loss, scale, margin):

    backbone = CNN_ARCHITECTURES[backbone](
        include_top=False,
        input_shape=[input_size, input_size, 3],
        weights='imagenet')

    pooling = GlobalGeMPooling2D(gem_p, name='head/gem', dtype='float32')

    dropout = tf.keras.layers.Dropout(
        dropout_rate, name='head/dropout')

    dense = tf.keras.layers.Dense(
        units=dense_units, name='head/dense')

    if loss == 'softmax':
        dense_output = tf.keras.layers.Dense(
            units=n_classes, name='head/vanilla')
    elif loss == 'arcface':
        margin = ArcMarginProduct(
            n_classes=n_classes, s=scale, m=margin,
            name='head/arc_margin', dtype='float32')
    else:
        margin = AddMarginProduct(
            n_classes=n_classes, s=scale, m=margin,
            name='head/cos_margin', dtype='float32')

    softmax = tf.keras.layers.Softmax(dtype='float32')

    image = tf.keras.layers.Input(
        [input_size, input_size, 3], name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)

    if loss == 'softmax':
        x = dense_output(tf.nn.relu(x))
    else:
        x = margin([x, label])

    x = softmax(x)

    return tf.keras.Model(
        inputs=[image, label], outputs=x)


class DistributedModel:

    def __init__(self, backbone, input_size, n_classes, phases,
                 batch_size, dense_units, dropout_rate, gem_p, loss,
                 scale, margin, clip_grad, checkpoint_weights,
                 optimizer, strategy, mixed_precision):

        self.model = create_model(
            backbone=backbone,
            input_size=None,
            n_classes=n_classes,
            gem_p=gem_p,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            loss=loss,
            scale=scale,
            margin=margin,)

        self.input_sizes = input_size
        self.batch_sizes = batch_size
        self.phases = phases

        if checkpoint_weights:
            self.model.load_weights(checkpoint_weights + '.h5')

        self.clip_grad = clip_grad

        self.optimizer = optimizer
        self.strategy = strategy
        self.mixed_precision = mixed_precision

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.mean_loss_train = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_top5_train = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, probs):
        per_example_loss = self.loss_object(labels, probs)
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
        with tf.GradientTape() as tape:
            probs = self.model(inputs, training=True)
            loss = self._compute_loss(inputs[1], probs)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
        self._backprop_loss(tape, loss, self.model.trainable_weights)
        self.mean_loss_train.update_state(inputs[1], probs)
        self.mean_accuracy_top5_train.update_state(inputs[1], probs)
        return loss

    @tf.function
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
                loss = self._distributed_train_step(inputs)
                train_ds.set_description(
                    "Phase {:01d}, Loss {:.3f}, Acc_5 {:.3f}".format(
                        phase,
                        self.mean_loss_train.result().numpy(),
                        self.mean_accuracy_top5_train.result().numpy(),
                    )
                )

            if save_path:
                self.model.save_weights(save_path + '.h5')

            self.mean_loss_train.reset_states()
            self.mean_accuracy_top5_train.reset_states()
