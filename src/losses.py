import tensorflow as tf



def angular_distance(features):
    features = tf.math.l2_normalize(features, axis=1)
    angular_distances = 1 - tf.matmul(features, features, transpose_b=True)
    return tf.maximum(angular_distances, 0.0)

def pairwise_distances(features, squared=False):
    dot_product = tf.matmul(features, tf.transpose(features))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)
    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances

def _masked_maximum(data, mask, dim=1):
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums

def _masked_minimum(data, mask, dim=1):
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums

def triplet_hard_loss(y_true, y_pred, margin=1.0, distance_metric='L2', soft=False):

    labels, embeddings = tf.squeeze(y_true), y_pred

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    labels = tf.expand_dims(labels, -1)

    if distance_metric == "L1":
        distance_matrix = pairwise_distances(
            precise_embeddings, squared=False
        )

    elif distance_metric == "L2":
        distance_matrix = pairwise_distances(
            precise_embeddings, squared=True
        )

    elif distance_metric == "angular":
        distance_matrix = angular_distance(precise_embeddings)

    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)
    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)

    # obtain the smallest Distance(A, N) for each feature
    hard_negatives = _masked_minimum(
        distance_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)
    mask_positives = adjacency - tf.linalg.diag(tf.ones([batch_size]))

    # obtain the largest Distance(A, P) for each feature
    hard_positives = _masked_maximum(
        distance_matrix, mask_positives)

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    triplet_loss = tf.reduce_mean(triplet_loss)

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss

def triplet_semihard_loss(y_true, y_pred, margin=1.0, distance_metric="L2"):

    labels, embeddings = tf.squeeze(y_true), y_pred

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    labels = tf.expand_dims(labels, -1)

    if distance_metric == "L1":
        distance_matrix = pairwise_distances(
            precise_embeddings, squared=False
        )
    elif distance_metric == "L2":
        distance_matrix = pairwise_distances(
            precise_embeddings, squared=True
        )
    elif distance_metric == "angular":
        distance_matrix = angular_distance(precise_embeddings)
    else:
        distance_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    distance_matrix_tiled = tf.tile(distance_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            distance_matrix_tiled, tf.reshape(tf.transpose(distance_matrix), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(distance_matrix_tiled, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(distance_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, distance_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss
