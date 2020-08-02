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

def create_model(backbone,
                 input_shape,
                 n_classes,
                 pretrained_weights=None,
                 dense_units=512,
                 dropout_rate=0.0,
                 regularization_factor=None,
                 loss='arcface',
                 scale=30,
                 margin=0.3):

    backbone = CNN_ARCHITECTURES[backbone](
        include_top=False,
        input_shape=input_shape,
        weights=pretrained_weights)

    #pooling = GlobalGeMPooling2D(initial_p=3., name='head/gem_pooling', dtype='float32')
    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(units=dense_units, name='head/dense')

    if loss == "arcface":
        margin = ArcMarginProduct(
            n_classes=n_classes,
            s=scale,
            m=margin,
            name='head/arc_margin',
            dtype='float32')
    else:
        # cosface
        margin = AddMarginProduct(
            n_classes=n_classes,
            s=scale,
            m=margin,
            name='head/cos_margin',
            dtype='float32')

    softmax = tf.keras.layers.Softmax(dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = margin([x, label])
    x = softmax(x)
    return tf.keras.Model(
        inputs=[image, label], outputs=x)


class DistributedModel:

    def __init__(self,
                 backbone,
                 input_size,
                 n_classes,
                 batch_size,
                 pretrained_weights,
                 finetuned_weights,
                 dense_units,
                 dropout_rate,
                 regularization_factor,
                 loss,
                 scale,
                 margin,
                 clip_grad,
                 optimizer,
                 strategy,
                 mixed_precision):

        self.model = create_model(
            backbone=backbone,
            input_shape=input_size,
            n_classes=n_classes,
            pretrained_weights=pretrained_weights,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            regularization_factor=regularization_factor,
            loss=loss,
            scale=scale,
            margin=margin,)

        self.input_size = input_size
        self.batch_size = batch_size

        if finetuned_weights:
            self.model.load_weights(finetuned_weights)

        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.clip_grad = clip_grad

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.mean_loss_train = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_train = tf.keras.metrics.SparseTopKCategoricalAccuracy(
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
        self.mean_accuracy_train.update_state(inputs[1], probs)
        return loss, probs

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss, per_replica_probs = self.strategy.run(
            self._train_step, args=(dist_inputs,))
        return (
            self.strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                per_replica_loss,
                axis=None),
            per_replica_probs
        )

    def _extract_per_replica_values(self, epoch, inputs, probs):
        per_replica_inputs = self.strategy.experimental_local_results(inputs)
        per_replica_probs = self.strategy.experimental_local_results(probs)
        # loop over replicates
        for inpts, probs in zip(per_replica_inputs, per_replica_probs):
            # loop over examples
            for prob, label, ID in zip(probs, inpts[1], inpts[-1]):
                conf = tf.gather(prob, label) + self.min_confidence
                ID = ID.numpy().decode('utf-8')
                self.confidences[ID] = tf.clip_by_value(
                    conf, 0.0, 1.0).numpy()

    def train(self, train_df, epochs, save_path):

        compute_confidence_from_epoch = 0
        self.min_confidence = 0.9

        # define accumulator(s)
        self.confidences = {}

        for epoch in range(epochs):

            if epoch >= compute_confidence_from_epoch:
                train_df['weight'] = (
                    np.where(train_df['id'].isin(self.confidences.keys()),
                             train_df['weight']*train_df['id'].map(self.confidences),
                             train_df['weight'])
                )
                self.min_confidence = max(self.min_confidence-0.1, 0.1)
                print('MINIMUM CONFIDENCE', self.min_confidence)
                print(train_df.head(10))

            train_ds = create_dataset(
                    dataframe=train_df,
                    training=True,
                    batch_size=self.batch_size,
                    input_size=self.input_size,
                )

            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            train_ds = tqdm.tqdm(train_ds)
            for i, inputs in enumerate(train_ds):
                loss, probs = self._distributed_train_step(inputs)
                train_ds.set_description(
                    "TRAIN: Loss {:.3f}, Accuracy {:.3f}".format(
                        self.mean_loss_train.result().numpy(),
                        self.mean_accuracy_train.result().numpy()
                    )
                )

                if epoch >= compute_confidence_from_epoch:
                    self._extract_per_replica_values(epoch, inputs, probs)

            if save_path:
                self.model.save_weights(save_path)

            self.mean_loss_train.reset_states()
            self.mean_accuracy_train.reset_states()
