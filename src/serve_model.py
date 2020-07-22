import tensorflow as tf
import numpy as np

from model import create_model
from generator import preprocess_input
from config import (
    config_1, config_2, config_3, config_4, config_5, config_6
)

from zipfile import ZipFile
import shutil
import os
import argparse
import sys
sys.path.append('../')

tf.config.set_visible_devices([], 'GPU')


class Serving:

    def __init__(self, configs, tta=10, export_dir='../output/served_models/'):
        self.configs = configs
        self.model = self._create_model_ensemble()
        self.feature_dim = sum(config['dense_units'] for config in self.configs)
        self.tta = tta
        self.export_dir = export_dir

    def _create_model_ensemble(self):
        # create model and load finetuned weights
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
        input_images = tf.zeros([0, 384, 384, 3])
        for config in self.configs:
            for _ in range(self.tta):
                input_image = preprocess_input(
                    input_image, config['input_size'][:2], config['pad_on_resize'])
                input_images = tf.concat([input_images, input_image[tf.newaxis]], axis=0)

        input_images = tf.split(input_images, len(self.configs), axis=0)

        outputs = self.model(input_images)
        if tf.is_tensor(outputs):
            features = tf.reduce_mean(outputs, axis=0)
        else:
            features = tf.concat([
                tf.reduce_mean(output, axis=0)
                for output in outputs], axis=0)
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
                print(fp, '/'.join(fp.split('/')[2:]))
                zip.write(fp, arcname='/'.join(fp.split('/')[2:]))

# @tf.function(input_signature=[
#     tf.TensorSpec(
#         shape=[None, None, 3],
#         dtype=tf.uint8,
#         name='input_image')
# ])
# def serving(input_image):
#
#     dim = 0
#     for config in CONFIGS:
#         dim += config['dense_units']
#
#     features = tf.zeros((dim,), dtype=tf.float32)
#     augmented_images = []
#     for k in range(8):
#         input_images = [
#             preprocess_input(input_image, config['input_size'][:2], config['pad_on_resize'])
#             for config in CONFIGS
#         ]
#         input_images = [
#             tf.expand_dims(input_image, 0, name=f'image_{i}/expand_dims')
#             for i, input_image in enumerate(input_images)
#         ]
#         outputs = model(input_images)
#         if tf.is_tensor(outputs):
#             features += tf.math.l2_normalize(outputs[0])
#         else:
#             features += tf.concat([
#                 tf.math.l2_normalize(output[0]) for output in outputs], axis=0)
#     return {
#         'global_descriptor': tf.identity(features, name='global_descriptor')
#     }
#
# def create_model_ensemble():
#     # create model and load finetuned weights
#     models = []
#     for config in CONFIGS:
#         model = create_model(
#             backbone=config['backbone'],
#             input_shape=config['input_size'],
#             n_classes=config['n_classes'],
#             pretrained_weights=config['pretrained_weights'],
#             dense_units=config['dense_units'],
#             dropout_rate=config['dropout_rate'],
#             regularization_factor=config['regularization_factor'],
#             loss=config['loss']['type'],
#             scale=config['loss']['scale'],
#             margin=config['loss']['margin'],
#         )
#         model.load_weights(config['save_path'])
#         models.append(model)
#     # remove margin layer from each model, and finally merge models
#     for i in range(len(models)):
#         models[i] = tf.keras.Model(
#             inputs=models[i].get_layer('input/image').input,
#             outputs=models[i].get_layer('head/dense').output)
#         for layer in models[i].layers:
#             layer._name = layer.name + f'_{i}'
#     return tf.keras.Model(
#         inputs=[m.input for m in models],
#         outputs=[m.output for m in models])
#
# model = create_model_ensemble()
# tf.saved_model.save(model, '../tmp/model', {'serving_default': serving})

if __name__ == '__main__':
    serving = Serving(configs=[config_1,], tta=10)
    serving.save()
    serving.zip()

# TESTING --------------------------------------------
# image = tf.io.read_file(
#     '../input/landmark-retrieval-2020/train/0/0/0/000a0aee5e90cbaf.jpg')
# image = tf.image.decode_jpeg(image, channels=3)
# model_loaded = tf.saved_model.load('../tmp/model')
# f = model_loaded.signatures["serving_default"]
# test1 = f(input_image=image)['global_descriptor']
#
# image = preprocess_input(
#     image, config['input_size'][:2], config['pad_on_resize'], augment=False)
# out = model([image[tf.newaxis,...], image[tf.newaxis,...]])
# out1 = tf.math.l2_normalize(out[0][0])
# out2 = tf.math.l2_normalize(out[1][0])
# test2 = tf.concat([out1, out2], axis=0)
# print('served model output =  ', test1.numpy()[:10])
# print('regular model output = ', test2.numpy()[:10])
# #assert (all(test1['global_descriptor'].numpy().astype(np.float32)
# #        == test2.numpy().astype(np.float32)))
# #print("Model successfully loaded!")
# -----------------------------------------------------

# OUTPUT ----------------------------------------------
# filepaths = []
# for dirpath, _, filepath in os.walk('../tmp/model'):
#     for fp in filepath:
#         filepaths.append(os.path.join(dirpath, fp))
#
# with ZipFile('../output/served_models/my_model.zip', 'w') as zip:
#     for fp in filepaths:
#         print(fp, '/'.join(fp.split('/')[2:]))
#         zip.write(fp, arcname='/'.join(fp.split('/')[2:]))

#shutil.rmtree('../tmp/model')
# ----------------------------------------------------
