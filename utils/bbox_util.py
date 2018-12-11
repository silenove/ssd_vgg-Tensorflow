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
