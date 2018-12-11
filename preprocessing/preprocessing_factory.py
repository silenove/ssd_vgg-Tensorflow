"""Contains a factory for building various models."""

import tensorflow as tf

from preprocessing import ssd_vgg_preprocess

slim = tf.contrib.slim

preprocessing_fn_map = {
    'ssd_vgg_300': ssd_vgg_preprocess
}


def get_preprocessing(name):
    """Return preprocessing_fn(image, labels, bboxes, ...)"""
    if name not in preprocessing_fn_map.keys():
        raise ValueError('Preprocessing name [%s] was not recognized.' % name)

    return preprocessing_fn_map[name].preprocess_image
