import tensorflow as tf
import numpy as np
from zipfile import ZipFile
import shutil
import os
import argparse
import sys
sys.path.append('../')

from model import create_model
from generator import preprocess_input
from config import (
    config_1, config_2, config_3, config_4, config_5, config_6
)

tf.config.set_visible_devices([], 'GPU')


class Serving:

    def __init__(self, configs, tta=10, export_dir='../output/served_models/'):
        self.configs = configs
        self.model = self._create_model_ensemble()
        self.feature_dim = sum(config['dense_units'] for config in self.configs)
        self.tta = tta
        self.export_dir = export_dir

    def _create_model_ensemble(self):
        models = []
        for i, config in enumerate(self.configs):
            model = create_model(
                backbone=config['backbone'],
                input_shape=config['input_size'],
                n_classes=config['n_classes'],
                pretrained_weights=config['pretrained_weights'],
                dense_units=config['dense_units'],
                dropout_rate=config['dropout_rate'],
                regularization_factor=config['regularization_factor'],
                loss=config['loss']['type'],
                scale=config['loss']['scale'],
                margin=config['loss']['margin'],
            )
            model.load_weights(config['save_path'])
            model = tf.keras.Model(
                inputs=model.get_layer('input/image').input,
                outputs=model.get_layer('head/dense').output)
            for layer in model.layers:
                layer._name = layer.name + '_' + str(i)
            models.append(model)
        return tf.keras.Model(
            inputs=[m.input for m in models],
            outputs=[m.output for m in models])

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, 3],
            dtype=tf.uint8,
            name='input_image')
    ])
    def _extract_global_descriptor(self, input_image):

        inputs = []
        for config in self.configs:
            input_images = []
            for scale in config['tt_scaling']:
                h = int(config['input_size'][0]*scale)
                w = int(config['input_size'][1]*scale)
                input_image = preprocess_input(
                    input_image, target_size=(h, w))
                input_images.append(input_image[tf.newaxis])
            inputs.append(input_images)

        features = tf.zeros([512*len(self.configs),], dtype='float32')
        for i in range(1):
            outputs = self.model([inp[i] for inp in inputs])
            if tf.is_tensor(outputs):
                features += tf.math.l2_normalize(outputs[0])
            else:
                features += tf.concat([out[0] for out in outputs], axis=0)

        return {
            'global_descriptor': tf.identity(features, name='global_descriptor')
        }

    def save(self):
        tf.saved_model.save(
            obj=self.model,
            export_dir=self.export_dir+'model',
            signatures={'serving_default': self._extract_global_descriptor})

    def zip(self):
        filepaths = []
        for dirpath, _, filepath in os.walk(self.export_dir+'model'):
            for fp in filepath:
                filepaths.append(os.path.join(dirpath, fp))

        with ZipFile(self.export_dir + 'model.zip', 'w') as zip:
            for fp in filepaths:
                print(fp, '/'.join(fp.split('/')[4:]))
                zip.write(fp, arcname='/'.join(fp.split('/')[4:]))


if __name__ == '__main__':
    serving = Serving(configs=[config_1,], tta=1)
    serving.save()
    serving.zip()
