import tensorflow as tf
from models.efficientnet_sync import EfficientNetB0
from tensorflow.keras.applications import ResNet50

from layers import GlobalGeMPooling2DLayer, CombinedMarginLayer

def create_model(input_shape,
                 num_classes,
                 pretrained_weights=None,
                 dense_units=512,
                 dropout_rate=0.0,
                 regularization_factor=1e-10,
                  # recommended if combined: [64/30, 1.00, 0.30, 0.20]
                  #                       or [64/30, 0.90, 0.40, 0.15]
                  margin_param={
                     's':  1.0,         # recommended: 64 or 30
                     'm1': 1.00,        # (sphere) recommended: 1.35
                     'm2': 0.10,        # (arc)    recommended: 0.50
                     'm3': 0.00,        # (cos)    recommended  0.35
                 }):

    backbone = ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=pretrained_weights)

    pooling = GlobalGeMPooling2DLayer(name='head/gem_pooling')

    batch_norm1 = tf.keras.layers.BatchNormalization()
    batch_norm2 = tf.keras.layers.BatchNormalization()
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(
        units=dense_units,
        kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
        name='head/dense')

    margin = CombinedMarginLayer(
        num_classes=num_classes,
        regularizer=tf.keras.regularizers.l2(regularization_factor),
        s =margin_param['s' ],
        m1=margin_param['m1'],
        m2=margin_param['m2'],
        m3=margin_param['m3'],
        name='head/margin',
        dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    x = pooling(x)
    x = batch_norm1(x)
    x = dropout(x)
    x = dense(x)
    x = batch_norm2(x)
    x = margin([x, label])

    return tf.keras.Model(
        inputs=[image, label], outputs=x)
