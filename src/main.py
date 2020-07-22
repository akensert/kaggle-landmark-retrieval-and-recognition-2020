import numpy as np
import tensorflow as tf
import math

from model import create_model, DistributedModel
from generator import create_triplet_dataset, create_singlet_dataset
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


dataset = create_singlet_dataset(
    input_path=config['input_path'],
    batch_size=config['batch_size'],
    input_size=config['input_size'],
    pad_on_resize=config['pad_on_resize'],
    K=config['K'],
    shuffle_buffer_size=100000)

with strategy.scope():

    optimizer = get_optimizer(
        opt=config['optimizer'],
        steps_per_epoch=math.ceil(800_000/config['batch_size']),
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
        finetuned_weights='../output/weights/model.h5',
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
        epochs=config['n_epochs'], ds=dataset, save_path=config['save_path'])
