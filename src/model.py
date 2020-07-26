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

BACKBONE_ZOO = {
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

    backbone = BACKBONE_ZOO[backbone](
        include_top=False,
        input_shape=input_shape,
        weights=pretrained_weights)

    pooling = GlobalGeMPooling2D(name='head/gem_pooling', dtype='float32')
    batch_norm = tf.keras.layers.experimental.SyncBatchNormalization(
        name='head/sync_batch_norm')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(
        units=dense_units,
        kernel_regularizer=None,
        name='head/dense')

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
    x = batch_norm(x)
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
                 optimizer,
                 strategy,
                 mixed_precision,
                 clip_grad=10.):

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
        self.global_batch_size = batch_size * strategy.num_replicas_in_sync

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
        self.mean_loss_valid = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_valid = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, probs):
        per_example_loss = self.loss_object(labels, probs)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=self.global_batch_size)

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

    def _predict_step(self, inputs):
        probs = self.model(inputs, training=False)
        self.mean_loss_valid.update_state(inputs[1], probs)
        self.mean_accuracy_valid.update_state(inputs[1], probs)
        return probs

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss = self.strategy.run(self._train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def _distributed_predict_step(self, dist_inputs):
        probs = self.strategy.run(self._predict_step, args=(dist_inputs,))
        if tf.is_tensor(probs): return [probs]
        else: return probs.values

    def train_and_eval(self, train_df, valid_df, epochs, save_path):
        for epoch in range(epochs):
            train_ds = create_dataset(
                    dataframe=train_df,
                    training=True,
                    batch_size=self.batch_size,
                    input_size=self.input_size,
                    K=1
                )
            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            train_ds = tqdm.tqdm(train_ds)
            for i, inputs in enumerate(train_ds):
                loss = self._distributed_train_step(inputs)
                train_ds.set_description(
                    "TRAIN: Loss {:.3f}, Accuracy {:.3f}".format(
                        self.mean_loss_train.result().numpy(),
                        self.mean_accuracy_train.result().numpy()
                    )
                )

            if valid_df is not None:
                valid_ds = create_dataset(
                        dataframe=valid_df,
                        training=False,
                        batch_size=4,
                        input_size=self.input_size,
                        K=1
                    )
                valid_ds = self.strategy.experimental_distribute_dataset(valid_ds)
                valid_ds = tqdm.tqdm(valid_ds)
                for inputs in valid_ds:
                    probs = self._distributed_predict_step(inputs)
                    valid_ds.set_description(
                        "VALID: Loss {:.3f}, Accuracy {:.3f}".format(
                            self.mean_loss_valid.result().numpy(),
                            self.mean_accuracy_valid.result().numpy()
                        )
                    )

            if save_path:
                self.model.save_weights(save_path)

            self.mean_loss_train.reset_states()
            self.mean_loss_valid.reset_states()
            self.mean_accuracy_train.reset_states()
            self.mean_accuracy_valid.reset_states()

    def _test(self, test_df):
        '''
        this method is only to be used with cleaning/fit_and_predict.py
        '''
        test_ds = create_dataset(
                dataframe=test_df,
                training=False,
                batch_size=1,
                input_size=self.input_size,
                K=1
            )
        test_ds = self.strategy.experimental_distribute_dataset(test_ds)
        test_ds = tqdm.tqdm(test_ds)

        out = np.zeros([0, 3], dtype=np.float32)
        for inputs in test_ds:
            probs = self._distributed_predict_step(inputs)
            for prob in probs:
                prob = tf.squeeze(prob)
                prob_target = tf.gather(prob, tf.squeeze(inputs[1])).numpy()
                prob_max = tf.reduce_max(prob).numpy()
                prob_min = tf.reduce_min(prob).numpy()
                out = np.concatenate(
                    [out, np.array([[prob_target, prob_max, prob_min]])],
                    axis=0
                )
        return out
