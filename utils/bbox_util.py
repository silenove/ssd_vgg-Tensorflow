"""Bounding boxes methods."""

import numpy as np
import tensorflow as tf


def bboxes_clip(bbox_ref, bboxes, scope=None):
    """Clip bounding boxes to a reference box.
    Arguments:
        bbox_ref: Reference bounding box. Nx4 or 4 shaped-Tensor.
        bboxes: Bounding boxes to clip. Nx4 or 4 shaped-Tensor.
    """
    with tf.name_scope(scope, 'bboxes_clip'):
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)

        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        # Double check!
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes


def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box. Useful for updating
    a collection of boxes after cropping an image.
    """
    with tf.name_scope(name, 'bboxes_resize'):
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def bboxes_sort_all_classes(dict_scores, dict_bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    Arguments:
        scores: Dictionary, item - batch_size x -1 Tensor float scores.
        bboxes: Dictionary, item - batch_size x -1 x 4 Tensor bounding boxes.
    """

    def gather(bboxes, idxes):
        bboxes_gather = tf.gather(bboxes, idxes)
        return bboxes_gather

    def bboxes_sort_one_class(scores, bboxes, top_k):
        scores_sorted, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)
        bboxes_sorted = tf.map_fn(lambda x: gather(x[0], x[1]),
                                  [bboxes, idxes],
                                  dtype=bboxes.dtype,
                                  parallel_iterations=10,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True)
        return scores_sorted, bboxes_sorted

    with tf.name_scope(scope, 'detected_bboxes_sort', values=[dict_scores, dict_bboxes]):
        dict_scores_sorted = {}
        dict_bboxes_sorted = {}
        for cls in dict_scores.keys():
            scores_sorted, bboxes_sorted = bboxes_sort_one_class(dict_scores[cls],
                                                                 dict_bboxes[cls],
                                                                 top_k)
            dict_scores_sorted[cls] = scores_sorted
            dict_bboxes_sorted[cls] = bboxes_sorted
        return dict_scores_sorted, dict_bboxes_sorted


def bboxes_nms_all_classes(dict_scores_sorted, dict_bboxes_sorted, batch_size,
                           nms_threshold=0.5, keep_top_k=200,
                           scope=None):
    """Apply non-maximum selection to bounding boxes.
    Arguments:
        dict_scores_sorted: Dictionary (class: scores), scores - batch x top_k.
        dict_bboxes_sorted: Dictionary (class: bboxes), bboxes - batch x top_k x 4.
    """

    def bboxes_nms_one_class(scores_sorted, bboxes_sorted, batch_size,
                             nms_threshold, keep_top_k):
        scores_batches = []
        bboxes_batches = []
        for i in range(batch_size):
            idxes = tf.image.non_max_suppression_padded(bboxes_sorted[i],
                                                        scores_sorted[i],
                                                        keep_top_k,
                                                        nms_threshold,
                                                        pad_to_max_output_size=True)
            scores = tf.gather(scores_sorted[i], idxes)
            bboxes = tf.gather(bboxes_sorted[i], idxes)

            scores_batches.append(scores)
            bboxes_batches.append(bboxes)
        scores_batches = tf.stack(scores_batches, axis=0)
        bboxes_batches = tf.stack(bboxes_batches, axis=0)
        return scores_batches, bboxes_batches

    with tf.name_scope(scope, 'bboxes_nms_all_classes'):
        dict_scores_nms = {}
        dict_bboxes_nms = {}
        for cls in dict_scores_sorted.keys():
            scores_nms, bboxes_nms = bboxes_nms_one_class(dict_scores_sorted[cls],
                                                          dict_bboxes_sorted[cls],
                                                          batch_size,
                                                          keep_top_k,
                                                          nms_threshold)
            dict_scores_nms[cls] = scores_nms
            dict_bboxes_nms[cls] = bboxes_nms
        return dict_scores_nms, dict_bboxes_nms


def bboxes_filter_overlap(labels, bboxes, threshold=0.5, assign_negative=False,
                          scope=None):
    """Filter out bounding boxes based on (relative) overlap with reference
    box [0, 0, 1, 1]. Remove completely bounding boxes, or assign negative labels
    to the one outside.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], dtype=bboxes.dtype), bboxes)
        mask = scores > threshold
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a collection of
    bounding boxes.
    Arguments:
        bbox_ref: (N, 4) of (4,) Tensor with reference bounding boxes.
        bboxes: (N, 4) Tensor, collection of bounding boxes.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # intersection
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tf.where(tf.greater(bboxes_vol, 0.), tf.divide(inter_vol, bboxes_vol),
                          tf.zeros_like(inter_vol), name='intersection')
        return scores
