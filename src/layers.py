import tensorflow as tf

class CombinedMarginLayer(tf.keras.layers.Layer):
    '''
    Combines SphereFace, ArcFace and CosFace into one layer:

        s * arccos(m1 * theta_y + m2) - m3

    s  = scale
    m1 = sphere margin
    m2 = arc margin
    m3 = cos margin

    References:
        https://arxiv.org/pdf/1801.07698.pdf
    '''

    def __init__(self,
                 num_classes,
                 regularizer,
                 s, m1, m2, m3,
                 **kwargs):
        super(CombinedMarginLayer, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def build(self, input_shape):
        super(CombinedMarginLayer, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.num_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=self.regularizer)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)

        X = tf.math.l2_normalize(X, axis=1)
        W = tf.math.l2_normalize(self.W, axis=0)

        original_target_logit = tf.matmul(X, W)

        original_target_logit = tf.clip_by_value(
            original_target_logit, -1.0 + 1e-7, 1.0 - 1e-7)

        theta = tf.math.acos(original_target_logit)
        #theta = tf.math.acos(original_target_logit)

        marginal_target_logit = tf.math.cos(
            self.m1*theta + self.m2) - self.m3

        y_onehot = tf.cast(
            tf.one_hot(y, depth=self.num_classes),
            dtype=original_target_logit.dtype
        )
        logit = (
            original_target_logit * (1 - y_onehot)
            + marginal_target_logit * y_onehot
        )
        logit *= self.s

        return tf.nn.softmax(logit)


class GlobalGeMPooling2DLayer(tf.keras.layers.Layer):
    '''
    Combines avg and max pooling, controlled by parameter p, which can be
    optionally be trainable by setting trainable to True.

    when p -> inf, this layer functions as max pooling
    when p = 0,    this layer functions as avg pooling

    References:
        https://arxiv.org/pdf/1711.02512.pdf
        https://arxiv.org/pdf/1811.00202.pdf
    '''
    def __init__(self, initial_p=1., trainable=False, name='head/gem_pooling'):
        super(GlobalGeMPooling2DLayer, self).__init__(name)

        self.p = tf.Variable(
            initial_value=initial_p,
            trainable=trainable,
            dtype=tf.float16 # float16 if mixed_precision is True
        )

    def call(self, inputs, **kwargs):
        inputs = tf.clip_by_value(
            inputs, clip_value_min=1e-7, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs
