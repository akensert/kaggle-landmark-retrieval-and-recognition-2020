import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import math


def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

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

@tf.function
def _spatial_transform(image,
                       rotation=5.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):


    dim = tf.gather(tf.shape(image), 0)
    xdim = dim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # list destination pixel indices
    x = tf.repeat(tf.range(dim//2, -dim//2,-1), dim)
    y = tf.tile(tf.range(-dim//2, dim//2), [dim])
    z = tf.ones([dim*dim], dtype='int32')
    idx = tf.stack([x,y,z])

    # rotate destination pixels onto origin pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    idx2 = tf.clip_by_value(idx2, -dim//2+xdim+1, dim//2)

    # find origin pixel values
    idx3 = tf.stack([dim//2-idx2[0,], dim//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    image = tf.reshape(d, [dim, dim, 3])
    return image

def _pixel_transform(image,
                     hue_delta=0.0,
                     saturation_delta=0.0,
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

def preprocess_input(image, target_size, augment):
    '''
    also for serving
    '''
    image = tf.image.resize(
        image, target_size, method='bilinear')
    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def read_data(input_path):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    data = pd.read_csv(input_path + 'train.csv')
    data['path'] = data['id'].map(mapping)
    return data

def create_triplet_dataset(input_path,
                           batch_size,
                           input_size,
                           K,
                           shuffle_buffer_size=1):


    def prepare_data(data):
        data = (
            data
            .groupby('landmark_id')['path']
            .agg(lambda x: ','.join(x)) # 'path1,path2,path3,path4,...'
            #.apply(list) # ['path1', 'path2', 'path3', 'path4', ...]
            .reset_index()
        )
        return data.path, data.index

    def sample_input(image_paths, label, K):
        image_paths = tf.strings.split(image_paths, sep=',')
        labels = tf.tile([label], [K,])
        if K-len(image_paths) > 0:
            image_paths = tf.random.shuffle(image_paths)
            for i in tf.range(K-len(image_paths)):
                image_paths = tf.concat(
                    [image_paths, [tf.gather(image_paths, i)]], axis=0)
            return image_paths, labels
        idx = tf.argsort(tf.random.uniform(tf.shape(image_paths)))
        idx = tf.gather(idx, tf.range(K))
        image_paths = tf.gather(image_paths, idx)
        return image_paths, labels

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    def reshape(x, y):
        x = tf.reshape(x, (-1, *input_size))
        y = tf.reshape(y, (-1,))
        return x, y

    def nested(x, y):
        return (tf.data.Dataset.from_tensor_slices((x, y))
                .map(lambda x, y: (read_image(x), y),
                    tf.data.experimental.AUTOTUNE)
                .map(lambda x, y: (preprocess_input(
                        x, input_size[:2], True), y),
                     tf.data.experimental.AUTOTUNE)
                .batch(K))

    dataframe = read_data(input_path)
    image_paths, labels = prepare_data(dataframe)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
        lambda x, y: sample_input(x, y, K), tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(nested)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: reshape(x, y), tf.data.experimental.AUTOTUNE)
    return dataset

def create_singlet_dataset(input_path,
                           batch_size,
                           input_size,
                           K,
                           shuffle_buffer_size=1):

    def prepare_data(data):
        alpha = 0.75
        counts_map = dict(
            data.groupby('landmark_id')['path'].agg(lambda x: len(x)))
        data['counts'] = data['landmark_id'].map(counts_map)
        data['probs'] = (
            (1/data.counts**alpha) / (1/data.counts**alpha).max())
        uniques = data['landmark_id'].unique()
        uniques_map = dict(zip(uniques, range(len(uniques))))
        data['labels'] = data['landmark_id'].map(uniques_map)
        return data.path, data.labels, data.probs.astype(np.float32)

    def filter_by_prob(x, y, p):
        if tf.random.uniform((), 0, 1) <= p:
            return True
        else:
            return False

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    dataframe = read_data(input_path)
    image_paths, labels, probs = prepare_data(dataframe)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, probs))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.filter(filter_by_prob)
    dataset = dataset.map(
        lambda x, y, p: (read_image(x), y), tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y: (preprocess_input(x, input_size[:2], True), y),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
