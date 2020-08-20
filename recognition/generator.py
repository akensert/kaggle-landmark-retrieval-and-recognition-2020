import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import math
import tqdm
import random

import augmentation


@tf.function
def load_image(image_path, central_crop=False, crop_ratio=(0.7, 1.0), dim=512):
    '''
    This functions takes an image path as input, reads the image, then central
    crops it or random crops it. Both type of croppings will keep the aspect ratio
    of the image, but only the random crop will function as a random zoom and
    a random shift (could be used as image augmentation/jitter). Finally, the
    cropped image is resized to (dim, dim, 3).

    Arguments:
        image_path: path to jpeg image (str)
        central_crop: if image should be central cropped (bool)
        crop_ratio: [if central_crop == False] crops the image randomly
            between 0.7*min_dim and 1.0*min_dim.
        dim: target size of the image, if dim = 512, final image size
            will be (512, 512, 3)
    Returns:
        image tensor: a cropped and resized image of size (dim, dim, 3)

    '''

    def random_truncated_normal_offset(max_offset, min_offset):
        '''
        Computes random offset for the cropping of an image, according
        to a truncated normal distribution.

        Note:
            stddev=max_offset/4 (of tf.random.normal()) will create
            sort of a rounded triangular distribution shape, while
            stddev=max_offset/6 will create more like a normal
            distribution shape, due to truncation.
        '''
        _ = tf.constant(-1, dtype='float32')
        offset = tf.while_loop(
            cond=lambda x: x<min_offset or x>max_offset,
            body=lambda x: tf.random.normal((), max_offset/2, max_offset/4),
            loop_vars=[_])
        return tf.cast(offset[0], dtype='int32')

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    shape = tf.shape(image)[:-1]


    if central_crop:
        if shape[0] > shape[1]:
            offset_height = (shape[0]-shape[1])//2
            offset_width = 0
            target_height = target_width = shape[1]
        else:
            offset_height = 0
            offset_width = (shape[1]-shape[0])//2
            target_height = target_width = shape[0]

        image_cropped=tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=target_height,
            target_width=target_width,
        )
        return tf.cast(
            tf.image.resize(
                image_cropped, (dim, dim), method='area'),
            dtype='uint8'
        )

    min_dim = tf.reduce_min(shape)
    crop_size = tf.cast(min_dim, 'float32') * tf.random.uniform((), *crop_ratio)

    y_max_offset = tf.cast(shape[0], dtype='float32')-crop_size
    x_max_offset = tf.cast(shape[1], dtype='float32')-crop_size
    min_offset = tf.constant(0, dtype='float32')

    offset_height = random_truncated_normal_offset(y_max_offset, min_offset)
    offset_width = random_truncated_normal_offset(x_max_offset, min_offset)
    target_height = target_width = tf.cast(crop_size, dtype='int32')

    image_cropped = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width,
    )
    return tf.cast(
            tf.image.resize(
                image_cropped, (dim, dim), method='area'),
            dtype='uint8'
        )

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image, label

def create_dataset(dataframe,
                   undersample=True,
                   batch_size=32,
                   target_dim=384,
                   central_crop=False,
                   crop_ratio=(0.7, 1.0),
                   apply_augmentation=False):

    if undersample:
        dataframe = dataframe.sample(
            frac=undersample, replace=False, weights='prob', axis=0)

    paths, labels = dataframe.path, dataframe.label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        lambda x, y: (load_image(x, central_crop, crop_ratio, target_dim), y),
        tf.data.experimental.AUTOTUNE
    )
    if apply_augmentation:
        dataset = dataset.map(
            lambda x, y: (augmentation.apply_random_jitter(x), y),
            tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.map(normalize, tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
