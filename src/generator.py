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
                       rotation=3.0,
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
                     hue_delta=0.0,
                     saturation_delta=0.3,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    if hue_delta > 0:
        image = tf.image.random_hue(
            image, hue_delta)
    if saturation_delta > 0:
        image = tf.image.random_saturation(
            image, 1-saturation_delta, 1+saturation_delta)
    if contrast_delta > 0:
        image = tf.image.random_contrast(
            image, 1-contrast_delta, 1+contrast_delta)
    if brightness_delta > 0:
        image = tf.image.random_brightness(
            image, brightness_delta)
    return image

def preprocess_input(image, target_size, ratio=-1, augment=False):
    '''
    also for serving
    '''
    if ratio == -1: # h / w
        image = tf.image.resize(
            image, target_size, method='bilinear', preserve_aspect_ratio=True)
    elif ratio > 1: # h > w
        h = tf.gather(target_size, 0)
        w = int(tf.cast(h, tf.float32) / ratio)
        image = tf.image.resize(
            image, (h, w), method='bilinear')
    elif ratio < 1: # w > h
        w = tf.gather(target_size, 1)
        h = int(tf.cast(w, tf.float32) * ratio)
        image = tf.image.resize(
            image, (h, w), method='bilinear')
    else: # h == w
        h = tf.gather(target_size, 0)
        w = tf.gather(target_size, 1)
        image = tf.image.resize(
            image, (h, w), method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def _prepare_df(df_orig, alpha=0.75):
    df = df_orig.copy()
    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['probs'] = (
        (1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    uniques_map = dict(zip(uniques, range(len(uniques))))
    df['labels'] = df['landmark_id'].map(uniques_map)
    df['image_target_ratio'] = df['image_target_ratio'].astype(np.float32)
    return df

def _group_shuffle_df(df_orig, batch_size, undersample=False):

    df = df_orig.copy()

    if undersample:
        df = df.sample(frac=undersample, replace=False, weights='probs', axis=0)
    else:
        df = df[df.landmark_id != 138982]
        df = df.sample(frac=1)

    groups_idx = [
        df.index[np.where(df.image_target_ratio_group == i)[0]]
        for i in range(6)
    ]
    N = [
        math.ceil(groups_idx[i].shape[0] / batch_size)
        for i in range(6)
    ]
    groups = [
        np.array_split(groups_idx[i], N[i]) for i in range(6)
    ]

    groups_flattened = [i for j in groups for i in j]
    mapping = {}
    for i, g in enumerate(groups_flattened):
        for j in g:
            mapping[j] = i

    df['batch'] = df.index.map(mapping)
    df['batch'] = df['batch'].astype(np.int32)
    df = df.sort_values(by='batch')

    groups = [df for _, df in df.groupby('batch')]
    random.shuffle(groups)
    groups_no_remainder = []
    for group in groups:
        if len(group) == batch_size:
            groups_no_remainder.append(group)
    df = pd.concat(groups_no_remainder)
    return df

def create_dataset(dataframe, training, batch_size, input_size, K=None):

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    df = _prepare_df(dataframe, alpha=0.75)

    if training:
        df = _group_shuffle_df(df, batch_size, undersample=False)

    print(df)

    image_paths, labels, ratio = df.path, df.labels, df.image_target_ratio

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, ratio))
    dataset = dataset.map(
        lambda x, y, r: (read_image(x), y, r),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y, r: (preprocess_input(x, input_size[:2], r, True), y),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
