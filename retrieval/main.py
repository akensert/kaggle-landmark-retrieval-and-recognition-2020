import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection

from model import create_model, DistributedModel
from generator import create_dataset
from optimizer import get_optimizer
from config import config_1 as config

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
mixed_precision = False
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    mixed_precision = True

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


def prepare_dataframe(df_orig, alpha=1.0):
    df = df_orig.copy()
    repl_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['weight'] = 1 / df['landmark_id'].map(repl_map).astype(np.float32)**alpha
    df['label'] = df['label'].astype(np.int32)
    df['image_target_ratio'] = df['image_target_ratio'].astype(np.float32)
    return df

dataframe = pd.read_csv('../input/modified_train.csv')
#dataframe = dataframe.iloc[::150]
dataframe = prepare_dataframe(dataframe, alpha=config['data_sampling']['alpha'])


with strategy.scope():

    optimizer = get_optimizer(
        opt=config['optimizer'],
        steps_per_epoch=config['learning_rate']['steps_per_epoch'],
        lr_max=config['learning_rate']['max'],
        lr_min=config['learning_rate']['min'],
        warmup_epochs=config['learning_rate']['warmup_epochs'],
        decay_epochs=config['learning_rate']['decay_epochs'],
        power=config['learning_rate']['power'],
    )

    dist_model = DistributedModel(
        backbone=config['backbone'],
        input_size=config['input_size'],
        n_classes=config['n_classes'],
        phases=config['phases'],
        batch_size=config['batch_size'],
        dense_units=config['dense_units'],
        dropout_rate=config['dropout_rate'],
        gem_p=config['gem_p'],
        loss=config['loss']['type'],
        scale=config['loss']['scale'],
        margin=config['loss']['margin'],
        clip_grad=config['clip_grad'],
        checkpoint_weights=config['checkpoint_weights'],
        optimizer=optimizer,
        strategy=strategy,
        mixed_precision=mixed_precision)

    dist_model.train(
        train_df=dataframe,
        epochs=config['n_epochs'],
        sample_frac=config['data_sampling']['frac'],
        save_path=config['save_path'])
