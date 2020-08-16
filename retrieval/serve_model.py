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
    config_1, config_2,
)

tf.config.set_visible_devices([], 'GPU')


class Serving:

    def __init__(self, configs, export_dir='../output/served_models/'):
        self.configs = configs
        self.model = self._create_model_ensemble()
        self.export_dir = export_dir

    def _create_model_ensemble(self):
        models = []

        for i, config in enumerate(self.configs):

            model = create_model(
                backbone=config['backbone'],
                input_size=None,
                n_classes=config['n_classes'],
                gem_p=config['gem_p'],
                dense_units=config['dense_units'],
                dropout_rate=config['dropout_rate'],
                loss=config['loss']['type'],
                scale=config['loss']['scale'],
                margin=config['loss']['margin'],
            )

            model.load_weights(config['load_path'] + '.h5')

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
        '''
        Pseudo code:

        features_all = []
        For each (input, model) in models
            features = []
            for each scale in scales
                feat = model(input*scale)
                feat = l2norm(feat)
                features.append(feat)
            features = mean(features, axis=0)
            features = l2norm(features)
            features_all.append(features)

        features_all = l2norm(features_all)
        return features_all
        '''

        IMAGE_SCALES = [1.00]

        inputs = []
        for config in self.configs:
            input_images = []
            for scale in IMAGE_SCALES:
                size = int(config['served_input_size']*scale)
                input_image_p = preprocess_input(input_image, target_size=size)
                input_images.append(input_image_p[tf.newaxis])
            inputs.append(input_images)

        features = tf.zeros([0, len(inputs), 1024], dtype='float32')
        for i in range(len(IMAGE_SCALES)):
            outputs = self.model([inp[i] for inp in inputs], training=False)

            if tf.is_tensor(outputs):
                features = tf.concat(
                    [features, tf.math.l2_normalize(outputs[tf.newaxis], -1)],
                    axis=0)
            else:
                outputs = tf.concat(outputs, axis=0)
                features = tf.concat(
                    [features, tf.math.l2_normalize(outputs, -1)[tf.newaxis]],
                    axis=0)
                #features = tf.math.l2_normalize(features, -1)

        features = tf.transpose(features, [1, 0, 2])
        features = tf.reduce_mean(features, axis=1)

        # if len(inputs) > 1: normalize features (axis=1)?
        features = tf.reshape(features, [-1])
        features = tf.math.l2_normalize(features, axis=0)

        return {
            'global_descriptor': tf.identity(features, name='global_descriptor')
        }

    def save_model(self):
        tf.saved_model.save(
            obj=self.model,
            export_dir=self.export_dir+'model',
            signatures={'serving_default': self._extract_global_descriptor})

    def zip_model(self):
        filepaths = []
        for dirpath, _, filepath in os.walk(self.export_dir+'model'):
            for fp in filepath:
                filepaths.append(os.path.join(dirpath, fp))

        with ZipFile(self.export_dir + 'model.zip', 'w') as zip:
            for fp in filepaths:
                print(fp, '/'.join(fp.split('/')[4:]))
                zip.write(fp, arcname='/'.join(fp.split('/')[4:]))


if __name__ == '__main__':
    serving = Serving(configs=[config_2, ])
    serving.save_model()
    serving.zip_model()
