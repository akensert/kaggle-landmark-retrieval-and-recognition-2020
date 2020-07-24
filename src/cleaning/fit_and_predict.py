import numpy as np
import tensorflow as tf
import math
from sklearn import model_selection
import pandas as pd
import sys
sys.path.append('../')

from model import create_model, DistributedModel
from generator import read_data, create_triplet_dataset, create_singlet_dataset
from optimizer import get_optimizer
from config import config_1 as config

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


dataframe = read_data('../' + config['input_path'])

sss = model_selection.StratifiedKFold(
    n_splits=2, shuffle=True, random_state=42
).split(X=dataframe.index, y=dataframe.landmark_id)

with strategy.scope():

    for i, (train_idx, valid_idx) in enumerate(sss):

        train_dataset = create_singlet_dataset(
            dataframe=dataframe.iloc[train_idx],
            training=True,
            batch_size=config['batch_size'],
            input_size=config['input_size'],
            K=config['K'])

        valid_dataset = create_singlet_dataset(
            dataframe=dataframe.iloc[valid_idx],
            training=False,
            batch_size=config['batch_size'],
            input_size=config['input_size'],
            K=config['K'])

        optimizer = get_optimizer(
            opt=config['optimizer'],
            steps_per_epoch=math.ceil(125_000/config['batch_size']),
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
            pretrained_weights=config['pretrained_weights'],
            finetuned_weights='../'+config['save_path'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            regularization_factor=config['regularization_factor'],
            loss=config['loss']['type'],
            scale=config['loss']['scale'],
            margin=config['loss']['margin'],
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=True)

        dist_model.train(
           epochs=16, ds=train_dataset, save_path="../"+config['save_path'])

        preds = dist_model.predict(ds=valid_dataset)
        pd.DataFrame(preds.astype(int), index=valid_idx).to_csv(
            f'../../output/predictions/preds_{i}.csv')
