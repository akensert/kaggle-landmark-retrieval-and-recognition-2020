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

        if finetuned_weights:
            self.model.load_weights(finetuned_weights)

        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy

        self.loss_metric = tf.keras.metrics.Mean()

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

        if not(os.path.isdir('../output/weights')) and save_best:
            os.makedirs('../output/weights')

    def _compute_loss(self, labels, probs):
        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, probs, from_logits=False, axis=-1
        )
        return tf.reduce_mean(per_example_loss) / self.strategy.num_replicas_in_sync

    @tf.function
    def _distributed_train_step(self, dist_inputs):

        def train_step(inputs):
            with tf.GradientTape() as tape:
                probs = self.model(inputs, training=True)
                loss = self._compute_loss(inputs[1], probs)
                self.loss_metric.update_state(loss)
                if self.mixed_precision:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)
            if self.mixed_precision:
                scaled_gradients = tape.gradient(
                    scaled_loss, self.model.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            return loss

        per_replica_loss = self.strategy.run(train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def _distributed_predict_step(self, dist_inputs):

        def predict_step(inputs):
            probs = self.model(inputs, training=False)
            return probs

        preds = self.strategy.run(predict_step, args=(dist_inputs,))
        if tf.is_tensor(preds):
            return [preds]
        else:
            return preds.values

    def predict(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds)
        ds = tqdm.tqdm(ds)

        preds_accum = np.zeros([0, 81313], dtype=np.float32)
        for inputs in ds:
            preds = self._distributed_predict_step(inputs)
            for pred in preds:
                preds_accum = np.concatenate([preds_accum, pred.numpy()], axis=0)

        return preds_accum

    def train(self, epochs, ds, save_path):
        for epoch in range(epochs):
            dataset = self.strategy.experimental_distribute_dataset(ds)
            dataset = tqdm.tqdm(dataset)
            for i, inputs in enumerate(dataset):
                loss = self._distributed_train_step(inputs)
                epoch_loss = self.loss_metric.result().numpy()
                dataset.set_description(
                    "Loss {:.4f}".format(
                        self.loss_metric.result().numpy(),
                    )
                )
            self.loss_metric.reset_states()
            if save_path:
                self.model.save_weights(save_path)
