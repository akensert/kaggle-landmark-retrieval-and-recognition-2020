import tensorflow as tf

from layers import GlobalGeMPooling2D, ArcMarginProduct, AddMarginProduct

_resnet_architectures = {
    'resnet-50':          tf.keras.applications.ResNet50,
    'resnet-101':         tf.keras.applications.ResNet101,
    'resnet-152':         tf.keras.applications.ResNet152,
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
            inputs=model.input, outputs=outputs, name=model.name)


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
        self.bn_conv2d_1 = tf.keras.layers.BatchNormalization(
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

        # # ResNet50
        # layers = {
        #     'conv4_block6_out': 'block4',
        #     'conv5_block3_out': 'block5'
        # }
        # # ResNet101
        # layers = {
        #     'conv4_block23_out':'block4',
        #     'conv5_block3_out': 'block5'
        # }
        # # ResNet152
        # layers = {
        #     'conv4_block36_out':'block4',
        #     'conv5_block3_out': 'block5'
        # }

        self.backbone = ExtractIntermediateLayers(
            model_object=_resnet_architectures['resnet-50'],
            input_dim=input_dim,
            layers={'conv4_block6_out':'block4',
                    'conv5_block3_out':'block5'}
        )
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.desc_fc = tf.keras.layers.Dense(dense_units)
        if margin_type == 'arcface':
            self.desc_margin = ArcMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')
            self.attn_margin = ArcMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')

        else:
            self.desc_margin = AddMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')
            self.attn_margin = AddMarginProduct(
                n_classes=81313, s=scale, m=margin, dtype='float32')

        self.attention = Attention()
        self.attn_fc = tf.keras.layers.Dense(dense_units)

        self.softmax = tf.keras.layers.Softmax(dtype='float32')

    def forward_prop_desc(self, images, labels, training=True):
        features = self.backbone(images, training=training)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
        x = self.desc_margin([x, labels])
        return self.softmax(x), features['block4']

    def forward_prop_attn(self, features, labels, training=True):
        x, _, _ = self.attention(features, training=training)
        x = self.attn_fc(x)
        x = self.attn_margin([x, labels])
        return self.softmax(x)

    @property
    def descriptor_weights(self):
        return (self.backbone.trainable_weights+
                self.desc_margin.trainable_weights+
                self.desc_fc.trainable_weights)

    @property
    def attention_weights(self):
        return (self.attention.trainable_weights+
                self.attn_margin.trainable_weights+
                self.attn_fc.trainable_weights)

    def call(self, inputs, training=False):
        '''
        Although it may be used for the training loop, this call method
        is used for model.build(...)-->model.load_weights(...) only.
        '''
        out1, block4 = self.forward_prop_desc(*inputs, training=training)
        out2 = self.forward_prop_attn(block4, inputs[1], training=training)
        return (out1, out2)
