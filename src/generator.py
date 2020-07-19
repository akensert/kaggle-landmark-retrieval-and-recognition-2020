import tensorflow as tf
import numpy as np


def preprocess_input(image, target_size, pad):
    '''
    Serving and offline
    '''
    if pad:
        image = tf.image.resize_with_pad(
            image, *target_size, method='bilinear')
    else:
        image = tf.image.resize(
            image, target_size, method='bilinear')
    image /= 255.
    return image

def create_dataset(dataframe,
                   batch_size,
                   input_size,
                   pad_on_resize,
                   K,
                   shuffle_buffer_size=1):

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
                        x, input_size[:2], pad_on_resize), y),
                     tf.data.experimental.AUTOTUNE)
                .batch(K))

    image_paths = dataframe.path.values
    labels = np.array(dataframe.index)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(lambda x, y: sample_input(x, y, K), tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(nested)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: reshape(x, y), tf.data.experimental.AUTOTUNE)

    return dataset
