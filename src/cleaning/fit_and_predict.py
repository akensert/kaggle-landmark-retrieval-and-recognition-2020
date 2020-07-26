import numpy as np
import tensorflow as tf
import math
from sklearn import model_selection
import pandas as pd
import sys
sys.path.append('../')

from model import create_model, DistributedModel
from optimizer import get_optimizer
from config import config_1 as config

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
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

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


dataframe = pd.read_csv('../../input/modified_train.csv')
dataframe['path'] = dataframe['path'].apply(lambda x: '../' + x)

sss = model_selection.StratifiedKFold(
    n_splits=2, shuffle=True, random_state=42
).split(X=dataframe.index, y=dataframe.landmark_id)

with strategy.scope():

    for i, (train_idx, test_idx) in enumerate(sss):

        train_dataset = dataframe.iloc[train_idx]
        test_dataset = dataframe.iloc[test_idx]

        optimizer = get_optimizer(
            opt=config['optimizer'],
            steps_per_epoch=math.ceil(500_000/config['batch_size']),
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
            batch_size=config['batch_size'],
            pretrained_weights=config['pretrained_weights'],
            finetuned_weights=None,
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            regularization_factor=config['regularization_factor'],
            loss=config['loss']['type'],
            scale=config['loss']['scale'],
            margin=config['loss']['margin'],
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=True)

        dist_model.train_and_eval(
            train_df=train_dataset, valid_df=None,
            epochs=8,
            save_path='../'+config['save_path'])

        output = dist_model._test(test_df=test_dataset)

        pd.DataFrame(output, columns=['target', 'max', 'min'], index=test_idx).to_csv(
            f'../../output/predictions/preds_{i}.csv')
