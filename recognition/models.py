import tensorflow as tf

from backbones import efficientnet_bn_sync as efficientnet
from backbones import resnet_bn_sync as resnet
from backbones import densenet_bn_sync as densenet
from backbones import inception_bn_sync as inception
from backbones import inceptionresnet_bn_sync as inceptionresnet
from backbones import xception_bn_sync as xception

from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct

CNN_ARCHITECTURES = {
    'efficientnet-b0':    efficientnet.EfficientNetB0,
    'efficientnet-b1':    efficientnet.EfficientNetB1,
    'efficientnet-b2':    efficientnet.EfficientNetB2,
    'efficientnet-b3':    efficientnet.EfficientNetB3,
    'efficientnet-b4':    efficientnet.EfficientNetB4,
    'efficientnet-b5':    efficientnet.EfficientNetB5,
    'efficientnet-b6':    efficientnet.EfficientNetB6,
    'efficientnet-b7':    efficientnet.EfficientNetB7,
    'resnet-50':          resnet.ResNet50,
    'resnet-101':         resnet.ResNet101,
    'resnet-152':         resnet.ResNet152,
    'densenet-121':       densenet.DenseNet121,
    'densenet-169':       densenet.DenseNet169,
    'densenet-201':       densenet.DenseNet201,
    'inception-v3':       inception.InceptionV3,
    'inceptionresnet-v2': inceptionresnet.InceptionResNetV2,
    'xception':           xception.Xception,
}


class ExtractIntermediateLayers:

    def __new__(cls, model_instance, layers):
        outputs = {}
        for layer in model_instance.layers:
            if layer.name in layers.keys():
                outputs[layers[layer.name]] = layer.output
        return tf.keras.Model(
            inputs=model_instance.input, outputs=outputs)


class AutoEncoder(tf.keras.Model):

    def __init__(self, name='AutoEncoder'):
        super(AutoEncoder, self).__init__(name=name)

    def call(self, ):
        pass


class Attention(tf.keras.Model):

    def __init__(self, input_dim, decay=1e-10, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=input_dim,
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

    def __init__(self, n_classes, s, m, input_dim=None,
                 backbone='resnet-50', **kwargs):
        super(Delf, self).__init__(**kwargs)

        self.backbone = ExtractIntermediateLayers(
            CNN_ARCHITECTURES[backbone](
                include_top=False,
                input_shape=[input_dim, input_dim, 3],
                weights='imagenet'),
            {'conv4_block6_out':'block4', 'conv5_block3_out':'block5'}
        )
        self.pooling = GlobalGeMPooling2D(
            initial_p=3., name='delf/gem', dtype='float32')

        self.attention = Attention(
            input_dim=self.backbone.output['block4'].shape[-1],
            name='Attention')

        self.desc_fc = tf.keras.layers.Dense(512, name='delf/desc_fc')
        self.attn_fc = tf.keras.layers.Dense(n_classes, name='delf/attn_fc')

        self.arc_margin = ArcMarginProduct(n_classes, s=s, m=m, dtype='float32')
        self.softmax = tf.keras.layers.Softmax(dtype='float32')


    def forward_prop_desc(self, images, labels, training=True):
        features = self.backbone(images, training=training)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
        x = self.arc_margin([x, labels])
        return self.softmax(x), features

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
