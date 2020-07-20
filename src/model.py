import tensorflow as tf
from models.efficientnet_sync import EfficientNetB0
from tensorflow.keras.applications import ResNet50

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

    backbone = ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=pretrained_weights)

    pooling = GlobalGeMPooling2D(name='head/gem_pooling')
    batch_norm1 = tf.keras.layers.experimental.SyncBatchNormalization()
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
    x = batch_norm1(x)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = batch_norm2(x)
    x = margin([x, label])

    return tf.keras.Model(
        inputs=[image, label], outputs=x)
