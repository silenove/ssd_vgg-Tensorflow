"""The utils needed in SSD implementations."""

import numpy as np
import tensorflow as tf


def encode_gtbboxes_from_one_layer(ssd_params,
                                   anchors_cord,
                                   gt_labels,
                                   gt_bboxes,
                                   matching_threshold=0.5,
                                   dtype=tf.float32):
    """Encode gt labels and bboxes using SSD anchors from one layer.
    Arguments:
        anchors_layer: the default anchors in one layer.
        gt_labels: 1D tensor(int64) containing gt labels.
        gt_bboxes: Nx4 tensor(float) with bboxes relative coordinates.
        matching_threshold: Threshold for positive match with gt bboxes.


    """

    y_ref, x_ref, h_ref, w_ref = anchors_cord
    ymin = y_ref - h_ref / 2.0
    ymax = y_ref + h_ref / 2.0
    xmin = x_ref - w_ref / 2.0
    xmax = x_ref + w_ref / 2.0
    anchors = [ymin, xmin, ymax, xmax]

    shape_ref = y_ref.shape
    anchor_labels = tf.zeros(shape_ref, dtype=tf.int64)
    anchor_scores = tf.zeros(shape_ref, dtype=dtype)

    target_ymin = tf.zeros(shape_ref, dtype=dtype)
    target_xmin = tf.zeros(shape_ref, dtype=dtype)
    target_ymax = tf.zeros(shape_ref, dtype=dtype)
    target_xmax = tf.zeros(shape_ref, dtype=dtype)

    def condition(i, anchor_labels, anchor_scores, target_ymin,
                  target_xmin, target_ymax, target_xmax):
        cond = tf.less(i, tf.shape(gt_labels))
        return cond[0]

    def body(i, anchor_labels, anchor_scores, target_ymin,
             target_xmin, target_ymax, target_xmax):
        label = gt_labels[i]
        bbox = gt_bboxes[i]
        jaccard = jaccard_between_anchors_and_gt(anchors, bbox)

        mask = tf.greater(jaccard, anchor_scores)
        mask = tf.logical_and(mask, tf.greater(anchor_scores, matching_threshold))
        mask = tf.logical_and(mask, label < ssd_params.num_classes)
        mask_int = tf.cast(mask, tf.int64)
        mask_float = tf.cast(mask, dtype)

        anchor_labels = mask_int * label + (1 - mask_int) * anchor_labels
        anchor_scores = tf.where(mask, jaccard, anchor_scores)

        target_ymin = mask_float * bbox[0] + (1 - mask_float) * target_ymin
        target_xmin = mask_float * bbox[1] + (1 - mask_float) * target_xmin
        target_ymax = mask_float * bbox[2] + (1 - mask_float) * target_ymax
        target_xmax = mask_float * bbox[3] + (1 - mask_float) * target_xmax

        return [i + 1, anchor_labels, anchor_scores, target_ymin, target_xmin,
                target_ymax, target_xmax]

    i = 0
    [i, anchor_labels, anchor_scores, target_ymin, target_xmin,
     target_ymax, target_xmax] = tf.while_loop(condition, body,
                                               [i, anchor_labels, anchor_scores,
                                                target_ymin, target_xmin,
                                                target_ymax, target_xmax])

    # Transform to center / size
    target_y = (target_ymin + target_ymax) / 2
    target_x = (target_xmin + target_xmax) / 2
    target_h = target_ymax - target_ymin
    target_w = target_xmax - target_xmin

    devi_y = (target_y - y_ref) / h_ref / ssd_params.prior_variance[0]
    devi_x = (target_x - x_ref) / w_ref / ssd_params.prior_variance[1]
    devi_h = tf.log(target_h / h_ref) / ssd_params.prior_variance[2]
    devi_w = tf.log(target_w / w_ref) / ssd_params.prior_variance[3]

    target_loc = tf.stack([devi_y, devi_x, devi_h, devi_w], axis=-1)
    return anchor_labels, anchor_scores, target_loc


def encode_gtbboxes_from_all_layers(ssd_params,
                                    anchors_cords,
                                    gt_labels,
                                    gt_bboxes,
                                    matching_threshold=0.5,
                                    dtype=tf.float32,
                                    scope='gtbboxes_encoder'):
    """Encode gt labels and bboxes using SSD anchors from all layers.

    Arguments:
        anchors_layer: the default anchors in one layer.
        gt_labels: 1D tensor(int64) containing gt labels.
        gt_bboxes: Nx4 tensor(float) with bboxes relative coordinates.
        matching_threshold: Threshold for positive match with gt bboxes.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_scores = []
        target_locs = []
        for layer in ssd_params.featmap_layers:
            with tf.name_scope('gtbboxes_encoder_%s' % layer):
                labels, scores, loc = encode_gtbboxes_from_one_layer(ssd_params,
                                                                     anchors_cords[layer],
                                                                     gt_labels,
                                                                     gt_bboxes,
                                                                     matching_threshold,
                                                                     dtype)
                target_labels.append(labels)
                target_scores.append(scores)
                target_locs.append(loc)
        return target_labels, target_scores, target_locs


def jaccard_between_anchors_and_gt(anchors, bbox):
    """Compute jaccard score between a box and anchors of one layer.
    Arguments:
        anchors: anchors coordinates of one layer:[ymin, xmin, ymax, xmax], shape: (4 x h x w x num_anchors)
        bbox: a box coordinates: [ymin, xmin, ymax, xmax], shape: (4,)
    """
    ymin, xmin, ymax, xmax = anchors
    inter_ymin = tf.minimum(ymin, bbox[0])
    inter_xmin = tf.minimum(xmin, bbox[1])
    inter_ymax = tf.maximum(ymax, bbox[2])
    inter_xmax = tf.maximum(xmax, bbox[3])
    inter_h = tf.maximum(inter_ymax - inter_ymin, 0.0)
    inter_w = tf.maximum(inter_xmax - inter_xmin, 0.0)

    vol_anchors = (ymax - ymin) * (xmax - xmin)
    vol_inter = tf.multiply(inter_h, inter_w)
    vol_union = vol_anchors - vol_inter + ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    jaccard = tf.div(vol_inter, vol_union)
    return jaccard


def abs_smooth_L1(x):
    if tf.less(tf.abs(x), 1):
        return 0.5 * tf.pow(x, 2)
    else:
        return tf.abs(x) - 0.5


def decode_bboxes_from_all_layer(locs_pred, anchors,
                                 prior_scaling=[0.1, 0.1, 0.2, 0.2],
                                 scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD predicted localization and
       default anchors.
    """
    with tf.name_scope(scope):
        y_ref, x_ref, h_ref, w_ref = anchors

        y = y_ref + locs_pred[:, :, 0] * h_ref * prior_scaling[0]
        x = x_ref + locs_pred[:, :, 1] * w_ref * prior_scaling[1]
        h = h_ref * tf.exp(locs_pred[:, :, 2] * prior_scaling[2])
        w = w_ref * tf.exp(locs_pred[:, :, 3] * prior_scaling[3])

        ymin = y - h / 2
        xmin = x - w / 2
        ymax = y + h / 2
        xmax = x + w / 2

        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def select_detected_bboxes_every_class(predictions, localizations, num_classes,
                                       select_threshold=0.01,
                                       scope='detected_bboxes_select_every_class'):
    """Select detected bboxes based on select_threshold, sort the detected bboxes
    in every class.
    Arguments:
        predictions: batch_size x -1 x num_classes Tensor.
        localization: batch_size x -1 x 4 Tensor.
    """
    with tf.name_scope(scope, values=[predictions, localizations]):
        dict_scores_filt = {}
        dict_bboxes_filt = {}
        for cls in range(1, num_classes):
            scores = predictions[:, :, cls]
            fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
            scores = scores * fmask
            bboxes = localizations * tf.expand_dims(fmask, axis=-1)

            dict_scores_filt[cls] = scores
            dict_bboxes_filt[cls] = bboxes
        return dict_scores_filt, dict_bboxes_filt


def select_detected_bboxes_all_classes(predictions, localizations,
                                     num_classes, select_threshold=0.01,
                                     scope='detected_bboxes_select_all_class'):
    """Select detected bboxes based on select_threshold, sort the detected bboxes
        in argmax class by predictions.
    Arguments:
        predictions: batch_size x -1 x num_classes Tensor.
        localization: batch_size x -1 x 4 Tensor.
    """
    with tf.name_scope(scope, values=[predictions, localizations]):
        # Compute the max-prediction class (except background)
        dict_scores_filt = {}
        dict_bboxes_filt = {}

        max_scores = tf.reduce_max(predictions[:, :, 1:], axis=-1)
        max_mask = tf.equal(predictions, tf.expand_dims(max_scores, axis=-1))
        predictions = predictions * tf.cast(max_mask, predictions.dtype)

        for cls in range(1, num_classes):
            scores = predictions[:, :, cls]
            fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
            scores = scores * fmask
            bboxes = localizations * tf.expand_dims(fmask, axis=-1)

            dict_scores_filt[cls] = scores
            dict_bboxes_filt[cls] = bboxes
        return dict_scores_filt, dict_bboxes_filt

