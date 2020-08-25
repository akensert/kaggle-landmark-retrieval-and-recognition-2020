import tensorflow as tf


def compute_receptive_boxes(h, w, rf, stride, padding):
    x, y = tf.meshgrid(tf.range(h), tf.range(w))
    coords = tf.reshape(tf.stack([y, x], axis=2), [-1, 2])
    point_boxes = tf.cast(tf.concat([coords, coords], axis=1), 'float32')
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    return stride * point_boxes + bias

def select_local_features(attention_probs,
                          features,
                          rf_boxes,
                          attention_threshold,
                          nms_max_feature_num,
                          nms_iou_threshold,):

    attention_probs = tf.reshape(attention_probs, [-1])
    features = tf.reshape(features, [-1, features.shape[-1]])
    indices = tf.reshape(tf.where(attention_probs >= attention_threshold), [-1])
    selected_boxes = tf.gather(rf_boxes, indices)
    selected_features = tf.gather(features, indices)
    selected_scores = tf.gather(attention_probs, indices)

    nms_max_feature_num = tf.minimum(nms_max_feature_num, len(selected_boxes))
    selected_indices = tf.image.non_max_suppression(
        boxes=selected_boxes,
        scores=selected_scores,
        max_output_size=nms_max_feature_num,
        iou_threshold=nms_iou_threshold)
    selected_boxes = tf.gather(selected_boxes, selected_indices)
    selected_features = tf.gather(selected_features, selected_indices)
    selected_scores = tf.gather(selected_scores, selected_indices)
    return selected_boxes, selected_features, selected_scores[:, tf.newaxis]

def compute_keypoint_centers(boxes):
    return tf.divide(
        tf.add(
            tf.gather(boxes, [0, 1], axis=1), tf.gather(boxes, [2, 3], axis=1)),
        2.0)
