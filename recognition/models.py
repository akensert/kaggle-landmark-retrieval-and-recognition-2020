import tensorflow as tf

from backbones import (
    efficientnet_bn_sync,
    resnet_bn_sync,
    densenet_bn_sync,
    inception_bn_sync,
    inceptionresnet_bn_sync,
    xception_bn_sync
)
from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct


_architectures = {
    'efficientnet-b0':    efficientnet_bn_sync.EfficientNetB0,
    'efficientnet-b1':    efficientnet_bn_sync.EfficientNetB1,
    'efficientnet-b2':    efficientnet_bn_sync.EfficientNetB2,
    'efficientnet-b3':    efficientnet_bn_sync.EfficientNetB3,
    'efficientnet-b4':    efficientnet_bn_sync.EfficientNetB4,
    'efficientnet-b5':    efficientnet_bn_sync.EfficientNetB5,
    'efficientnet-b6':    efficientnet_bn_sync.EfficientNetB6,
    'efficientnet-b7':    efficientnet_bn_sync.EfficientNetB7,
    'resnet-50':          resnet_bn_sync.ResNet50,
    'resnet-101':         resnet_bn_sync.ResNet101,
    'resnet-152':         resnet_bn_sync.ResNet152,
    'densenet-121':       densenet_bn_sync.DenseNet121,
    'densenet-169':       densenet_bn_sync.DenseNet169,
    'densenet-201':       densenet_bn_sync.DenseNet201,
    'inception-v3':       inception_bn_sync.InceptionV3,
    'inceptionresnet-v2': inceptionresnet_bn_sync.InceptionResNetV2,
    'xception':           xception_bn_sync.Xception,
}


class ExtractIntermediateLayers:

    def __new__(cls, model_object, input_dim, layers):
        model = model_object(
            include_top=False,
            input_shape=[input_dim, input_dim, 3],
            weights='imagenet'
        )
        outputs = {}
        for layer in model.layers:
            if layer.name in layers.keys():
                outputs[layers[layer.name]] = layer.output
        return tf.keras.Model(
            inputs=model.input, outputs=outputs)


class AutoEncoder(tf.keras.Model):

    def __init__(self, name='AutoEncoder'):
        super(AutoEncoder, self).__init__(name=name)

    def call(self, ):
        pass


class Attention(tf.keras.Model):

    def __init__(self, decay=1e-4, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(decay),
            padding='same',
            name='attention/conv2d_1')
        self.bn_conv2d_1 = tf.keras.layers.experimental.SyncBatchNormalization(
            axis=3, name='attention/bn_conv2d_1')
        self.relu_conv2d_1 = tf.keras.layers.Activation('relu')

        self.conv2d_2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(decay),
            padding='same',
            name='attention/conv2d_2')
        self.softplus_conv2d_2 = tf.keras.layers.Activation('softplus')

    def call(self, inputs, training=True):
        x = self.conv2d_1(inputs)
        x = self.bn_conv2d_1(x, training=training)
        x = self.relu_conv2d_1(x)

        score = self.conv2d_2(x)
        prob = self.softplus_conv2d_2(score)

        inputs = tf.math.l2_normalize(inputs, axis=-1)
        feat = tf.reduce_mean(
            tf.multiply(inputs, prob), [1, 2], keepdims=False)
        return feat, prob, score


class Delf(tf.keras.Model):

    def __init__(self, global_units, n_classes, p, s, m, input_dim=None,
                 backbone='resnet-50', **kwargs):
        super(Delf, self).__init__(**kwargs)

        self.input_dim = input_dim

        self.backbone = ExtractIntermediateLayers(
            model_object=_architectures[backbone],
            input_dim=input_dim,
            layers={
                'conv4_block6_out':'block4',
                'conv5_block3_out':'block5'
            }
        )
        self.pooling = GlobalGeMPooling2D(initial_p=p, dtype='float32')
        self.attention = Attention()
        self.desc_fc = tf.keras.layers.Dense(global_units)
        self.attn_fc = tf.keras.layers.Dense(n_classes)
        self.arc_margin = ArcMarginProduct(n_classes, s=s, m=m, dtype='float32')
        self.softmax = tf.keras.layers.Softmax(dtype='float32')


    def forward_prop_desc(self, images, labels, training=True):
        features = self.backbone(images, training=training)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
        x = self.arc_margin([x, labels])
        return self.softmax(x), features['block4']

    def forward_prop_attn(self, features, training=True):
        x, _, _ = self.attention(features, training=training)
        x = self.attn_fc(x)
        return self.softmax(x)

    @property
    def get_descriptor_weights(self):
        return (self.backbone.trainable_weights+
                self.arc_margin.trainable_weights+
                self.desc_fc.trainable_weights)

    @property
    def get_attention_weights(self):
        return (self.attention.trainable_weights +
                self.attn_fc.trainable_weights)

    def call(self, inputs, training=False):
        '''
        Although it may be used for the training loop, this call method
        is used for model.build(...)-->model.load_weights(...) only.
        '''
        out1, block4 = self.forward_prop_desc(*inputs, training=training)
        out2 = self.forward_prop_attn(block4, training=training)
        return (out1, out2)
