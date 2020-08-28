import tensorflow as tf

from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct

_architectures = {
    'efficientnet-b0':    tf.keras.applications.EfficientNetB0,
    'efficientnet-b1':    tf.keras.applications.EfficientNetB1,
    'efficientnet-b2':    tf.keras.applications.EfficientNetB2,
    'efficientnet-b3':    tf.keras.applications.EfficientNetB3,
    'efficientnet-b4':    tf.keras.applications.EfficientNetB4,
    'efficientnet-b5':    tf.keras.applications.EfficientNetB5,
    'efficientnet-b6':    tf.keras.applications.EfficientNetB6,
    'efficientnet-b7':    tf.keras.applications.EfficientNetB7,
    'resnet-50':          tf.keras.applications.ResNet50,
    'resnet-101':         tf.keras.applications.ResNet101,
    'resnet-152':         tf.keras.applications.ResNet152,
    'densenet-121':       tf.keras.applications.DenseNet121,
    'densenet-169':       tf.keras.applications.DenseNet169,
    'densenet-201':       tf.keras.applications.DenseNet201,
    'inception-v3':       tf.keras.applications.InceptionV3,
    'inceptionresnet-v2': tf.keras.applications.InceptionResNetV2,
    'xception':           tf.keras.applications.Xception,
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
    '''
    TODO
    '''
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

    def __init__(self,
                 dense_units,
                 margin_type,
                 scale,
                 margin,
                 input_dim,
                 **kwargs):

        super(Delf, self).__init__(**kwargs)

        self.input_dim = input_dim

        self.backbone = ExtractIntermediateLayers(
            model_object=_architectures['resnet-50'],
            input_dim=input_dim,
            layers={'conv4_block6_out':'block4',
                    'conv5_block3_out':'block5'}
        )
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.desc_fc = tf.keras.layers.Dense(dense_units)
        if margin_type == 'arcface':
            self.margin = ArcMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')
        else:
            self.margin = AddMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')

        self.attention = Attention()
        self.attn_fc = tf.keras.layers.Dense(81313)

        self.softmax = tf.keras.layers.Softmax(dtype='float32')

    def forward_prop_desc(self, images, labels, training=True):
        features = self.backbone(images, training=training)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
        x = self.margin([x, labels])
        return self.softmax(x), features['block4']

    def forward_prop_attn(self, features, training=True):
        x, _, _ = self.attention(features, training=training)
        x = self.attn_fc(x)
        return self.softmax(x)

    @property
    def get_descriptor_weights(self):
        return (self.backbone.trainable_weights+
                self.margin.trainable_weights+
                self.desc_fc.trainable_weights)

    @property
    def get_attention_weights(self):
        return (self.attention.trainable_weights+
                self.attn_fc.trainable_weights)

    def call(self, inputs, training=False):
        '''
        Although it may be used for the training loop, this call method
        is used for model.build(...)-->model.load_weights(...) only.
        '''
        out1, block4 = self.forward_prop_desc(*inputs, training=training)
        out2 = self.forward_prop_attn(block4, training=training)
        return (out1, out2)
