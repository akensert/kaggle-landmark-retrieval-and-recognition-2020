import tensorflow as tf
import os
from zipfile import ZipFile

import config
import generator
import models
import extraction


class ServedModel(models.Delf):

    def __init__(self,
                 dense_units,
                 margin_type,
                 scale,
                 margin,
                 input_dim):

        # initialize delf model
        super(ServedModel, self).__init__(
            dense_units, margin_type, scale, margin, input_dim)

        # build delf model (so that weights can be loaded)
        super(ServedModel, self).__call__([
            tf.ones((1, input_dim, input_dim, 3), dtype='float32'),
            tf.ones((1,), dtype='int32'),
        ])
        # load finetuned delf weights
        self.load_weights('../output/weights/' + self.backbone.name + '.h5')

    @tf.function(
        input_signature=[
            tf.TensorSpec([], tf.string),
            tf.TensorSpec([], tf.string),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.bool),
            tf.TensorSpec([2,], tf.float32)])
    def extract_image(self, path, image_id, dim, central_crop, crop_ratio):
        image_path = (
            path + '/' +
            tf.strings.substr(image_id, 0, 1) + '/' +
            tf.strings.substr(image_id, 1, 1) + '/' +
            tf.strings.substr(image_id, 2, 1) + '/' +
            image_id + '.jpg'
        )
        image = generator.load_image(
            image_path, dim, central_crop, crop_ratio)
        return generator.normalize(image)[tf.newaxis]

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32),
            tf.TensorSpec([], tf.bool)])
    def extract_global_descriptor(self, image, l2_norm):
        features = self.backbone(image, training=False)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
        if l2_norm:
            x = tf.nn.l2_normalize(x, axis=1)
        return tf.squeeze(x)

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32)])
    def extract_global_prediction(self, image):
        predictions, _ = self.forward_prop_desc(image, -1, training=False)
        return tf.squeeze(predictions)

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32),
            tf.TensorSpec([], tf.bool)])
    def extract_global_descriptor_and_prediction(self, image, l2_norm):
        features = self.backbone(image, training=False)
        x1 = self.pooling(features['block5'])
        x1 = self.desc_fc(x1)
        if l2_norm:
            x2 = tf.nn.l2_normalize(x1, axis=1)
        else:
            x2 = tf.identity(x1)
        x1 = self.desc_margin([x1, -1])
        return tf.squeeze(x2), tf.squeeze(self.softmax(x1))


    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32),
            tf.TensorSpec([], tf.bool),
            tf.TensorSpec([], tf.bool),
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec([], tf.int32)])
    def extract_local_descriptor(self,
                                 image,
                                 l2_norm,
                                 use_rf_boxes,
                                 attention_threshold,
                                 max_num_features):
        features = self.backbone(image, training=False)['block4']
        _, attention_probs, _ = self.attention(features, training=False)
        # resnet152: 1315, 16, 655
        # resnet101: 835,  16, 415,
        # resnet50:  291,  16, 143
        if use_rf_boxes:
            # current only have rf + stride + padding for resnet50/101/152
            boxes = extraction.compute_receptive_boxes(
                *features[0].shape[:2], rf=1315.0, stride=16, padding=655.0)
        else:
            boxes = extraction.compute_boxes(*features[0].shape[:2])

        boxes, feats, scores = extraction.select_local_features(
            attention_probs=attention_probs,
            features=features,
            boxes=boxes,
            attention_threshold=attention_threshold,
            max_num_features=max_num_features)
        if l2_norm:
            feats = tf.nn.l2_normalize(feats, axis=1)
        points = extraction.compute_keypoint_centers(boxes)
        return feats, points

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32)])
    def extract_local_prediction(self, image):
        features = self.backbone(image, training=False)['block4']
        return tf.squeeze(
            self.forward_prop_attn(features, -1, training=False))

    def save(self, path, and_zip=True):
        # save model
        tf.saved_model.save(obj=self, export_dir=path)
        # zip saved model
        file_paths = []
        for dir_path, _, file_path in os.walk(path):
            for fp in file_path:
                file_paths.append(os.path.join(dir_path, fp))
        with ZipFile(path + '.zip', 'w') as z:
            for file_path in file_paths:
                print(file_path, '/'.join(file_path.split('/')[4:]))
                z.write(file_path, arcname='/'.join(file_path.split('/')[4:]))


if __name__ == '__main__':

    import logging
    tf.get_logger().setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")

    tf.config.set_visible_devices([], 'GPU')

    model = ServedModel(
        config.config['dense_units'],
        config.config['loss']['type'],
        config.config['loss']['scale'],
        config.config['loss']['margin'],
        config.config['input_dim'])

    print("Saving model...")
    model.save('../output/served_models/model')
