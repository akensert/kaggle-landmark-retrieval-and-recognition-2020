import tensorflow as tf
import numpy as np

from model import create_model
from generator import preprocess_input
from config import config

from zipfile import ZipFile
import shutil
import os
import argparse
import sys
sys.path.append('../')

tf.config.set_visible_devices([], 'GPU')

# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', '-W', type=str,
                    default='../output/weights/model.h5')
parser.add_argument('--output_filepath', '-O', type=str,
                    default='../output/served_models/my_model')

args = parser.parse_args()

weights_path = args.weights
output_filepath = args.output_filepath


@tf.function(input_signature=[
    tf.TensorSpec(
        shape=[None, None, 3],
        dtype=tf.uint8,
        name='input_image')
])
def serving(input_image):
    input_image = preprocess_input(
        input_image, config['input_size'][:2], config['pad_on_resize'])
    input_image = tf.expand_dims(
       input_image, 0, name='image/expand_dims')
    features = model(input_image)[0]
    features = tf.math.l2_normalize(features)
    return {
        'global_descriptor': tf.identity(features, name='global_descriptor')
    }

model = create_model(
    input_shape=config['input_size'],
    n_classes=config['n_classes'],
    dense_units=config['dense_units'],
    dropout_rate=config['dropout_rate'],
    regularization_factor=config['regularization_factor'],
    loss=config['loss']['type'],
    scale=config['loss']['scale'],
    margin=config['loss']['margin'],
)

print("Loading weights ...")
test1 = model.trainable_variables[0][0][0][0]
model.load_weights(weights_path)
test2 = model.trainable_variables[0][0][0][0]
assert not(any(test1==test2))
print("... sucessfully!")

model = tf.keras.Model(
    inputs=model.get_layer('input/image').input,
    outputs=model.get_layer('head/dense').output
)

tf.saved_model.save(model, '../tmp/model', {'serving_default': serving})

# TESTING --------------------------------------------
image = tf.io.read_file(
    '../input/landmark-retrieval-2020/train/0/0/0/000a0aee5e90cbaf.jpg')
image = tf.image.decode_jpeg(image, channels=3)
model_loaded = tf.saved_model.load('../tmp/model')
f = model_loaded.signatures["serving_default"]
test1 = f(input_image=image)['global_descriptor']

image = preprocess_input(
    image, config['input_size'][:2], config['pad_on_resize'])
test2 = model(image[tf.newaxis,...])[0]
test2 = tf.math.l2_normalize(test2)
print('served model output =  ', test1.numpy()[:10])
print('regular model output = ', test2.numpy()[:10])
#assert (all(test1['global_descriptor'].numpy().astype(np.float32)
#        == test2.numpy().astype(np.float32)))
#print("Model successfully loaded!")
# -----------------------------------------------------

# OUTPUT ----------------------------------------------
filepaths = []
for dirpath, _, filepath in os.walk('../tmp/model'):
    for fp in filepath:
        filepaths.append(os.path.join(dirpath, fp))

with ZipFile(f'{output_filepath}.zip','w') as zip:
    for fp in filepaths:
        print(fp, '/'.join(fp.split('/')[2:]))
        zip.write(fp, arcname='/'.join(fp.split('/')[2:]))

#shutil.rmtree('../tmp/model')
# ----------------------------------------------------
