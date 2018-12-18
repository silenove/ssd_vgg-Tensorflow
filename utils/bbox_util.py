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
            idxes, _ = tf.image.non_max_suppression_padded(bboxes_sorted[i],
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
                                                          nms_threshold,
                                                          keep_top_k)
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

def bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection of
    bounding boxes.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)

        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        union_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) + \
                    (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1]) - \
                    inter_vol
        jaccard = tf.where(tf.greater(union_vol, 0.), tf.divide(inter_vol, union_vol),
                          tf.zeros_like(inter_vol), name='jaccard')
        return jaccard


def bboxes_matching(label, scores, bboxes, labels_gt, bboxes_gt,
                    difficults_gt, matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values, single-inputs."""
    with tf.name_scope(scope, 'bboxes_matching_single', [scores, bboxes, labels_gt,
                                                         bboxes_gt, difficults_gt]):
        total_size = tf.size(scores)
        label = tf.cast(label, labels_gt.dtype)
        difficults_gt = tf.cast(difficults_gt, tf.bool)
        num_bboxes_gt = tf.count_nonzero(tf.logical_and(tf.equal(label, labels_gt),
                                                        tf.logical_not(difficults_gt)))
        matching_gt = tf.zeros(tf.shape(labels_gt), dtype=tf.bool)
        range_gt = tf.range(tf.size(labels_gt), dtype=tf.int32)

        # True/False positive matching TensorArrays
        tensorarray_tp = tf.TensorArray(tf.bool, size=total_size, dynamic_size=False,
                                     infer_shape=True)
        tensorarray_fp = tf.TensorArray(tf.bool, size=total_size, dynamic_size=False,
                                     infer_shape=True)

        # Loop
        def condition(i, ta_tp, ta_fp, matching):
            r = tf.less(i, total_size)
            return r

        def body(i, ta_tp, ta_fp, matching_gt):
            # Jaccard score with gt bboxes
            bbox = bboxes[i]
            jaccard = bboxes_jaccard(bbox, bboxes_gt)
            jaccard = jaccard * tf.cast(tf.equal(label, labels_gt), jaccard.dtype)
            max_idx = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            max_jaccard = jaccard[max_idx]
            match = max_jaccard > matching_threshold
            is_exist = matching_gt[max_idx]
            not_difficult = tf.logical_not(difficults_gt[max_idx])

            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(is_exist)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(tf.logical_not(match), is_exist))
            ta_fp = ta_fp.write(i, fp)

            mask = tf.logical_and(tf.equal(range_gt, max_idx),
                                  tf.logical_and(not_difficult, match))
            matching_gt = tf.logical_or(matching_gt, mask)

            return [i+1, ta_tp, ta_fp, matching_gt]

        i = 0
        [i, tensorarray_tp, tensorarray_fp, matching_gt] = tf.while_loop(
            condition, body, [i, tensorarray_tp, tensorarray_fp, matching_gt],
            parallel_iterations=1, back_prop=False
        )

        tp_match = tf.reshape(tensorarray_tp.stack(), tf.shape(scores))
        fp_match = tf.reshape(tensorarray_fp.stack(), tf.shape(scores))

        return num_bboxes_gt, tp_match, fp_match



def bboxes_matching_batch(labels, scores, bboxes,
                          labels_gt, bboxes_gt, difficults_gt,
                          matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values, batched-inputs."""
    with tf.name_scope(scope, 'bboxes_matching_batch', [scores, bboxes, labels_gt,
                                                        bboxes_gt, difficults_gt]):
        dict_num_bboxes_gt = {}
        dict_tp = {}
        dict_fp = {}
        for label in labels:
            n, tp, fp = tf.map_fn(
                lambda x: bboxes_matching(label, x[0], x[1], x[2], x[3], x[4], matching_threshold),
                (scores[label], bboxes[label], labels_gt, bboxes_gt, difficults_gt),
                dtype=(tf.int64, tf.bool, tf.bool),
                parallel_iterations=10,
                back_prop=False,
                swap_memory=True,
                infer_shape=True,
            )
            dict_num_bboxes_gt[label] = n
            dict_tp[label] = tp
            dict_fp[label] = fp
        return dict_num_bboxes_gt, dict_tp, dict_fp


