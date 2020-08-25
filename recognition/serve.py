import tensorflow as tf
import os
from zipfile import ZipFile

import config
import generator
import models
import extraction

class ServedModel(models.Delf):

    def __init__(
        self, weights, global_units, n_classes, p, s, m, input_dim, backbone):

        # initialize delf model
        super(ServedModel, self).__init__(
            global_units, n_classes, p, s, m, input_dim, backbone)

        # build delf model (so that weights can be loaded)
        super(ServedModel, self).__call__([
            tf.ones((1, input_dim, input_dim, 3), dtype='float32'),
            tf.ones((1,), dtype='int32'),
        ])
        # load finetuned delf weights
        self.load_weights(weights)

    @tf.function(
        input_signature=[
            tf.TensorSpec([], tf.string),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.bool),
            tf.TensorSpec([2,], tf.float32)])
    def extract_image(self, image_path, dim, central_crop, crop_ratio):
        image = generator.load_image(
            image_path, dim, central_crop, crop_ratio)
        return generator.normalize(image)[tf.newaxis]

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32)])
    def extract_global_descriptor(self, image):
        features = self.backbone(image, training=False)
        x = self.pooling(features['block5'])
        x = self.desc_fc(x)
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
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.float32)])
    def extract_local_descriptor(self,
                                 image,
                                 attention_threshold,
                                 nms_max_feature_num,
                                 nms_iou_threshold):
        features = self.backbone(image, training=False)['block4']
        _, attention_probs, _ = self.attention(features, training=False)

        rf_boxes = extraction.compute_receptive_boxes(
            *features[0].shape[:2], rf=291.0, stride=16, padding=143.0)
        boxes, feats, scores = extraction.select_local_features(
            attention_probs=attention_probs,
            features=features,
            rf_boxes=rf_boxes,
            attention_threshold=attention_threshold,
            nms_max_feature_num=nms_max_feature_num,
            nms_iou_threshold=nms_iou_threshold)
        points = extraction.compute_keypoint_centers(boxes)
        return feats, points

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 3], tf.float32)])
    def extract_local_prediction(self, image):
        features = self.backbone(image, training=False)['block4']
        return tf.squeeze(
            self.forward_prop_attn(features, training=False))

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
    import argparse
    import glob
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")

    tf.config.set_visible_devices([], 'GPU')


    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    model = ServedModel(
        config.config['finetuned_weights'],
        config.config['dense_units'],
        config.config['n_classes'],
        config.config['gem_p'],
        config.config['loss']['scale'],
        config.config['loss']['margin'],
        config.config['input_dim'],
        config.config['backbone'])

    print("Saving model...")
    model.save('../output/served_models/model')

    if args.test:
        # test loading
        print("Loading model...")
        imported = tf.saved_model.load('../output/served_models/model')

        image = imported.extract_image(
            glob.glob('../input/' + 'train/0/0/0/*')[0],
            dim=config.config['input_dim'],
            central_crop=True,
            crop_ratio=(0.7, 1.0),
        )

        print("Global descriptor =\n")
        global_desc = imported.extract_global_descriptor(image)
        print(global_desc)
