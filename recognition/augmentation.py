import tensorflow as tf
import math

# _policy = {
#     'fliplr':     {'p': 0.5},
#     'downscale':  {'range': (0.5, 1.0),  'p': 0.25},
#     'compression':{'range': (85, 100),   'p': 0.0},
#     'brightness': {'range': (0.8, 1.2),  'p': 0.50},
#     'contrast':   {'range': (0.9, 1.1),  'p': 0.25},
#     'gamma':      {'range': (0.9, 1.1),  'p': 0.25},
#     'saturation': {'range': (0.5, 1.5),  'p': 0.50},
#     'hue':        {'range': (0.0, 0.1),  'p': 0.45},
#     'rotation':   {'range': (-10., 10.), 'p': 0.35},
#     'shear':      {'range': (-10., 10.), 'p': 0.35},
#     'scale':      {'range': (0.8, 1.2),  'p': 0.25},
#     'shift':      {'range': (-32., 32.), 'p': 0.0},
#     'noise':      {'range': (0., 10.),   'p': 0.25},
#     'cutout':     {'range': (4, 16), 'size': (32, 33), 'p': 0.0}
# }

_policy = {
    'fliplr':     {'p': 0.0},
    'downscale':  {'range': (0.75, 1.0), 'p': 0.0},
    'compression':{'range': (85, 100),   'p': 0.0},
    'brightness': {'range': (0.8, 1.2),  'p': 0.5},
    'contrast':   {'range': (0.9, 1.1),  'p': 0.5},
    'gamma':      {'range': (0.9, 1.1),  'p': 0.0},
    'saturation': {'range': (0.75, 1.75),'p': 0.5},
    'hue':        {'range': (0.0, 0.1),  'p': 0.0},
    'rotation':   {'range': (-10., 10.), 'p': 0.5},
    'shear':      {'range': (-10., 10.), 'p': 0.5},
    'scale':      {'range': (0.8, 1.2),  'p': 0.5},
    'shift':      {'range': (-32., 32.), 'p': 0.0},
    'noise':      {'range': (0., 10.),   'p': 0.0},
    'cutout':     {'range': (4, 16), 'size': (32, 33), 'p': 0.0}
}


def random_fliplr(image):
    return tf.reverse(image, [1])

def random_downscale(image, downscale_range=(0.5, 1.0)):
    scale = tf.random.uniform((), *downscale_range)
    orig_shape = tf.shape(image)[:-1]
    shape = tf.cast(orig_shape, 'float32') * scale
    shape = tf.cast(shape, 'int32')
    downscaled = tf.image.resize(image, shape, method='area')
    upscaled = tf.image.resize(downscaled, orig_shape, method='area')
    return tf.cast(upscaled, 'uint8')

def random_jpeg_compression(image, compression_range=(85, 100)):
    return tf.image.random_jpeg_quality(image, *compression_range)

def random_brightness(image, brightness_range=(0.8, 1.2)):
    delta = brightness_range[1]-brightness_range[0]
    return tf.image.random_brightness(image, delta)

def random_contrast(image, contrast_range=(0.9, 1.1)):
    return tf.image.random_contrast(image, *contrast_range)

def random_gamma(image, gamma_range=(0.9, 1.1)):
    gamma = tf.random.uniform((), *gamma_range)
    return tf.image.adjust_gamma(image, gamma)

def random_saturation(image, saturation_range=(0.5, 1.5)):
    return tf.image.random_saturation(image, *saturation_range)

def random_hue(image, hue_range=(0.0, 0.1)):
    max_delta = hue_range[1] - hue_range[0]
    return tf.image.random_hue(image, max_delta)

def random_spatial_transform(image,
                             rotation_range=(-10, 10),
                             shear_range=(-10, 10),
                             scale_range=(0.8, 1.2),
                             shift_range=(-0, 0)):

    def get_transform_matrix(rotation, shear, scale, shift):

        def get_3x3_mat(lst):
            return tf.reshape(tf.concat([lst], axis=0), [3,3])

        # convert degrees to radians
        rotation = math.pi * rotation / 360.
        shear    = math.pi * shear    / 360.
        yscale   = scale[0]
        xscale   = scale[1]
        yshift   = shift[0]
        xshift   = shift[1]

        one  = tf.constant([1],dtype='float32')
        zero = tf.constant([0],dtype='float32')

        c1   = tf.math.cos(rotation)
        s1   = tf.math.sin(rotation)
        rot_mat = get_3x3_mat([c1,    s1,   zero ,
                               -s1,   c1,   zero ,
                               zero,  zero, one ])

        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)
        shear_mat = get_3x3_mat([one,  s2,   zero ,
                                 zero, c2,   zero ,
                                 zero, zero, one ])

        zoom_mat = get_3x3_mat([one/yscale, zero,      zero,
                                zero,      one/xscale, zero,
                                zero,      zero,      one])

        shift_mat = get_3x3_mat([one,  zero, yshift,
                                 zero, one,  xshift,
                                 zero, zero, one   ])

        return tf.matmul(
            tf.matmul(rot_mat, shear_mat),
            tf.matmul(zoom_mat, shift_mat)
        )

    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = tf.random.uniform([1], *rotation_range)
    shear = tf.random.uniform([1],    *shear_range)
    scale = tf.random.uniform([2, 1], *scale_range)
    shift = tf.random.uniform([2, 1], *shift_range)

    m = get_transform_matrix(rotation, shear, scale, shift)

    # origin pixels
    y = tf.repeat(tf.range(ydim//2, -ydim//2,-1), xdim)
    x = tf.tile(tf.range(-xdim//2, xdim//2), [ydim])
    z = tf.ones([ydim*xdim], dtype='int32')
    idx = tf.stack([y, x, z])

    # destination pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    # clip to origin pixels range
    idx2y = tf.clip_by_value(idx2[0,], -ydim//2+yxdim+1, ydim//2)
    idx2x = tf.clip_by_value(idx2[1,], -xdim//2+xxdim+1, xdim//2)
    idx2 = tf.stack([idx2y, idx2x, idx2[2,]])

    # apply destinations pixels to image
    idx3 = tf.stack([ydim//2-idx2[0,], xdim//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    image = tf.reshape(d, [ydim, xdim, 3])
    return image

def random_gaussian_noise(image, noise_range=(0, 10)):
    image = tf.cast(image, 'float32')
    sigma = tf.random.uniform((), *noise_range)
    noise = tf.random.normal(tf.shape(image), 0, sigma)
    image = image + noise
    return tf.cast(tf.clip_by_value(image, 0, 255), 'uint8')

def random_cutout(image, num_box_range=(4, 16),
                  box_size_range=(32, 33), box_val=127):

    num = tf.random.uniform((), *num_box_range, dtype='int32')

    y, x = tf.shape(image)[0], tf.shape(image)[1]

    yoffset = y - box_size_range[-1]
    xoffset = x - box_size_range[-1]

    coords = tf.cast(
        tf.random.uniform([num, 2], [0, 0], [yoffset, xoffset]),
        'int32')

    sizes = tf.random.uniform([num, 2], *box_size_range, dtype='int32')

    boxes = tf.concat([coords, sizes], axis=-1)
    masks = tf.zeros([0, y, x], dtype=image.dtype)
    for box in boxes:
        padding_dims = [
            [box[0], y-(box[0]+box[2])],
            [box[1], x-(box[1]+box[3])]
        ]

        mask = tf.pad(
            tf.zeros((box[2], box[3]), dtype=image.dtype),
            padding_dims,
            constant_values=1,
        )
        image = tf.where(
            tf.expand_dims(mask, -1) == 0,
            tf.ones_like(image, dtype=image.dtype) * box_val,
            image)

    return image

def apply_random_jitter(image):

    if _policy['fliplr']['p'] > tf.random.uniform(()):
        image = random_fliplr(image)

    if _policy['downscale']['p'] > tf.random.uniform(()):
        image = random_downscale(image, _policy['downscale']['range'])

    if _policy['compression']['p'] > tf.random.uniform(()):
        image = random_jpeg_compression(image, _policy['compression']['range'])

    if _policy['brightness']['p'] > tf.random.uniform(()):
        image = random_brightness(image, _policy['brightness']['range'])

    if _policy['contrast']['p'] > tf.random.uniform(()):
        image = random_contrast(image, _policy['contrast']['range'])

    if _policy['gamma']['p'] > tf.random.uniform(()):
        image = random_gamma(image, _policy['gamma']['range'])

    if _policy['saturation']['p'] > tf.random.uniform(()):
        image = random_saturation(image, _policy['saturation']['range'])

    if _policy['hue']['p'] > tf.random.uniform(()):
        image = random_hue(image, _policy['hue']['range'])

    rotation = (-0.0, 0.0)
    shear = (-0.0, 0.0)
    scale = (1.0, 1.0)
    shift = (-0.0, 0.0)

    if _policy['rotation']['p'] > tf.random.uniform(()):
        rotation = _policy['rotation']['range']

    if _policy['shear']['p'] > tf.random.uniform(()):
        shear = _policy['shear']['range']

    if _policy['scale']['p'] > tf.random.uniform(()):
        scale = _policy['scale']['range']

    if _policy['shift']['p'] > tf.random.uniform(()):
        shift = _policy['shift']['range']

    image = random_spatial_transform(image, rotation, shear, scale, shift)

    if _policy['noise']['p'] > tf.random.uniform(()):
        image = random_gaussian_noise(image, _policy['noise']['range'])

    if _policy['cutout']['p'] > tf.random.uniform(()):
        image = random_cutout(
            image, _policy['cutout']['range'], _policy['cutout']['size'])

    return image
