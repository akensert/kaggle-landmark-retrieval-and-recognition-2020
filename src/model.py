import tensorflow as tf
import os
import tqdm
from models.efficientnet_sync import EfficientNetB0
from tensorflow.keras.applications import ResNet50, ResNet101

from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct

def create_model(input_shape,
                 n_classes,
                 pretrained_weights=None,
                 dense_units=512,
                 dropout_rate=0.0,
                 regularization_factor=None,
                 loss='arcface',
                 scale=30,
                 margin=0.3):

    backbone = EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights=pretrained_weights)

    pooling = GlobalGeMPooling2D(name='head/gem_pooling')
    # batch_norm1 = tf.keras.layers.experimental.SyncBatchNormalization()
    batch_norm2 = tf.keras.layers.experimental.SyncBatchNormalization()
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

    image = tf.keras.layers.Input(input_shape, name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    # x = batch_norm1(x)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = batch_norm2(x)
    x = margin([x, label])

    return tf.keras.Model(
        inputs=[image, label], outputs=x)


class DistributedModel:

    def __init__(self,
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

    def _compute_loss(self, labels, logits):
        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True, axis=-1
        )
        return tf.reduce_mean(per_example_loss) / self.strategy.num_replicas_in_sync

    @tf.function
    def _distributed_train_step(self, dist_inputs):

        def train_step(inputs):
            with tf.GradientTape() as tape:
                logits = self.model(inputs, training=True)
                loss = self._compute_loss(inputs[1], logits)
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

    def train(self, epochs, ds):
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
            self.model.save_weights('../output/weights/model.h5')
