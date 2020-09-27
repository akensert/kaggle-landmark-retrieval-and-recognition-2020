import tensorflow as tf


def compute_receptive_boxes(h, w, rf, stride, padding):
    x, y = tf.meshgrid(tf.range(h), tf.range(w))
    coords = tf.reshape(tf.stack([y, x], axis=2), [-1, 2])
    point_boxes = tf.cast(tf.concat([coords, coords], axis=1), 'float32')
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    return stride * point_boxes + bias

def compute_boxes(h, w):
    x, y = tf.meshgrid(tf.range(h), tf.range(w))
    coords = tf.reshape(tf.stack([y, x], axis=2), [-1, 2])
    point_boxes = tf.cast(tf.concat([coords, coords], axis=1), 'float32')
    return point_boxes

def select_local_features(attention_probs,
                          features,
                          boxes,
                          attention_threshold,
                          max_num_features):

    attention_probs = tf.reshape(attention_probs, [-1])
    features = tf.reshape(features, [-1, features.shape[-1]])
    indices = tf.reshape(tf.where(attention_probs >= attention_threshold), [-1])
    selected_boxes = tf.gather(boxes, indices)
    selected_features = tf.gather(features, indices)
    selected_scores = tf.gather(attention_probs, indices)

    if max_num_features != -1:
        indices = tf.argsort(selected_scores, direction='DESCENDING')
        indices = tf.gather(indices, tf.range(max_num_features))
        selected_boxes = tf.gather(selected_boxes, indices)
        selected_features = tf.gather(selected_features, indices)
        selected_scores = tf.gather(selected_scores, indices)

    return selected_boxes, selected_features, selected_scores[:, tf.newaxis]

def compute_keypoint_centers(boxes):
    return tf.divide(
        tf.add(
            tf.gather(boxes, [0, 1], axis=1), tf.gather(boxes, [2, 3], axis=1)),
        2.0)
