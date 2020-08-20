import pandas as pd
import numpy as np
import tqdm
import glob
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU') # Turn off GPU

def obtain_image_shape(image_path):
    '''
    Using TensorFlow ops to read, decode, and compute shape of the image.
    Can be exchanged for PIL.Image.open().shape or cv2.imread().shape.
    '''
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image.numpy().shape[:2]

def modify_dataframe(input_path):
    '''
    This function only uses a single thread. It takes approximately 30-90
    minutes to run this function. Because this modification of the csv file is
    only made once, a multi-thread version hasn't been implemented.
    '''
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path[3:]
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)

    image_shape = []
    image_ratio = []
    for i, row in tqdm.tqdm(df.iterrows()):
        shape = obtain_image_shape('../'+row.path)
        image_shape.append(shape)
        image_ratio.append(shape[0]/shape[1])

    df['image_shape'] = image_shape
    df['image_ratio'] = image_ratio
    df['image_target_ratio'] = pd.cut(
        x=df.image_ratio,
        bins=[-np.inf]+[0.6, 0.7, 0.8, 1.1, 1.4, 1.6]+[np.inf],
        labels=[0.563, 0.667, 0.750, 1.000, 1.3334, 1.501, 1.800]
    )
    df['image_target_ratio_group'] = df['image_target_ratio'].map(
        dict(
            zip(
                [0.563, 0.667, 0.750, 1.000, 1.3334, 1.501, 1.800],
                [0, 1, 2, 3, 4, 5, 6]
            )
        )
    )

    uniques = df['landmark_id'].unique()
    mapping = dict(zip(uniques, range(len(uniques))))
    df['label'] = df['landmark_id'].map(mapping)

    df.to_csv(
        '/'.join(input_path.split('/')[:-2])+'/'+modified_train.csv',
        index=False
    )


if __name__ == '__main__':

    modify_dataframe('../../input/')
