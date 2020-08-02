import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import math
import tqdm
import random

pd.options.mode.chained_assignment = None


def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 360.
    shear    = math.pi * shear    / 360.

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

    zoom_mat = get_3x3_mat([one/hzoom, zero,      zero,
                            zero,      one/wzoom, zero,
                            zero,      zero,      one])

    shift_mat = get_3x3_mat([one,  zero, hshift,
                             zero, one,  wshift,
                             zero, zero, one   ])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )

def _spatial_transform(image,
                       rotation=5.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):


    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    zoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    hzoom = zoom
    wzoom = zoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

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

def _pixel_transform(image,
                     saturation_delta=0.5,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    image = tf.image.random_saturation(
        image, 1-saturation_delta, 1+saturation_delta)
    image = tf.image.random_contrast(
        image, 1-contrast_delta, 1+contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image

def _random_fliplr(image, p=0.25):
    r = tf.random.uniform(())
    mirror_cond = tf.math.less(r, p)
    image = tf.cond(
        mirror_cond,
        lambda: tf.reverse(image, [1]),
        lambda: image
    )
    return image

def preprocess_input(image, target_size, ratio=-1, augment=False):
    '''
    also for serving
    '''
    if ratio == -1: # ratio = h / w
        h, w = tf.gather(target_size, 0), tf.gather(target_size, 1)
        image = tf.image.resize(
            image, (h, w), method='area', preserve_aspect_ratio=True)
    else:
        if ratio > 1: # h > w
            h = tf.gather(target_size, 0)
            w = int(tf.cast(h, tf.float32) / ratio)
        elif ratio < 1: # w > h
            w = tf.gather(target_size, 1)
            h = int(tf.cast(w, tf.float32) * ratio)
        else: # h == w
            h = tf.gather(target_size, 0)
            w = tf.gather(target_size, 1)

        image = tf.image.resize(image, (h, w), method='area')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        # image = _random_fliplr(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def _group_shuffle_df(df_orig, batch_size, undersample=False):

    df = df_orig.copy()

    if undersample:
        df = df.sample(
            frac=undersample, replace=False, weights='weight', axis=0)
    else:
        #df = df[df.landmark_id != 138982]
        df = df.sample(frac=1)

    groups_idx = [
        df.index[np.where(df.image_target_ratio_group == i)[0]]
        for i in range(7)
    ]
    N = [
        math.ceil(groups_idx[i].shape[0] / batch_size)
        for i in range(7)
    ]
    groups = [
        np.array_split(groups_idx[i], N[i]) for i in range(7)
    ]

    groups_flattened = [i for j in groups for i in j]
    del groups
    mapping = {}
    for i, g in enumerate(groups_flattened):
        for j in g:
            mapping[j] = i
    del groups_flattened
    df['batch'] = df.index.map(mapping)
    df.sort_values(by='batch', inplace=True)

    groups = [df for _, df in df.groupby('batch')]
    random.shuffle(groups)
    groups_no_remainder = []
    for group in groups:
        if len(group) == batch_size:
            groups_no_remainder.append(group)
    return pd.concat(groups_no_remainder)

def create_dataset(dataframe, training, batch_size, input_size):

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    df = _group_shuffle_df(
        dataframe, batch_size, undersample=0.25 if training else False)

    paths, labels, ratios, ids = df.path, df.label, df.image_target_ratio, df.id

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, ratios, ids))
    dataset = dataset.map(
        lambda x,y,r,i: (read_image(x), y, r, i),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x,y,r,i: (preprocess_input(x, input_size[:2], r, training), y, i),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
