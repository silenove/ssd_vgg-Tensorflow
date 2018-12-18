"""Preprocess images and bounding boxes for detection."""

import functools
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from utils import bbox_util

slim = tf.contrib.slim

BBOX_CROP_OVERLAP = 0.5
MIN_OBJECT_COVERED = 0.25

EVAL_SIZE = (300, 300)


def tf_summary_image(image, bboxes, name='image'):
    """Add image with bounding boxes to summary."""
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_bbox = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_bbox)


def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
    """Normalizes pixel values in the image.
    Move the pixel values from [original_minval, original_maxval] range to
    [target_minval, target_maxval].
    """
    with tf.name_scope('NormalizeImage', values=[image]):
        original_minval = float(original_minval)
        original_maxval = float(original_maxval)
        target_minval = float(target_minval)
        target_maxval = float(target_maxval)
        image = tf.to_float(image)
        image = tf.subtract(image, original_minval)
        image = tf.multiply(image, (target_maxval - target_minval) /
                            (original_maxval - original_minval))
        image = tf.add(image, target_minval)
        return image


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel) with sel sampled from [0, ..., num_cases - 1]"""
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([func(
        control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes."""

    def flip_bboxes(bboxes):
        """Filp bounding boxes coordinates."""
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    with tf.name_scope('random_flip_left_right'):
        cond = tf.less(tf.random_uniform([], 0., 1., seed=seed), 0.5)
        image = tf.cond(cond, lambda: tf.image.flip_left_right(image), lambda: image)
        bboxes = tf.cond(cond, lambda: flip_bboxes(bboxes), lambda: bboxes)
        return image, bboxes


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a tensor image.
    Arguments:
        image: 3-D Tensor containing single image in [0,1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
            else:
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.25)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.25)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.8, upper=1.25)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.25)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.25)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0., 1.)


def _rgb_to_grayscale(images, name=None):
    """Converts one or more images from RGB to Grayscale."""
    with tf.name_scope(name, 'rgb_to_grayscale', [images]) as name:
        images = tf.convert_to_tensor(images, name='images')
        orig_dtype = images.dtype
        flt_image = tf.image.convert_image_dtype(images, tf.float32)

        rgb_weights = [0.2989, 0.5870, 0.1140]
        rank_1 = tf.expand_dims(tf.rank(images) - 1, 0)
        gray_float = tf.reduce_sum(flt_image * rgb_weights, rank_1, keep_dims=True)
        gray_float.set_shape(images.get_shape()[:-1].concatenate([1]))
        return tf.image.convert_image_dtype(gray_float, orig_dtype, name=name)


def random_rgb_to_gray(image, probability=0.1, seed=None):
    """Changes the image from RGB to Grayscale with the given probability.
    Arguments:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
        with pixel values varying between [0, 255].
        probability: the probability of returning a grayscale image. [0, 1]
    """

    def _image_to_gray(image):
        image_gray1 = _rgb_to_grayscale(image)
        image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
        return image_gray3

    with tf.name_scope('RandomRGBtoGray', values=[image]):
        do_gray = tf.random_uniform([], minval=0., maxval=1., seed=seed)
        image = tf.cond(tf.greater(do_gray, probability), lambda: image,
                        lambda: _image_to_gray(image))
        return image


def distorted_bounding_box_crop(image, labels, bboxes, bbox=None,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    Arguments:
        image: 3-D Tensor of image (it will be converted to float in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax]. If num_boxes is 0  then it would use the whole image.
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        if bbox is None:
            bbox = tf.constant([0., 0., 1., 1.], dtype=tf.float32, shape=[1, 1, 4])
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True
        )
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, image.get_shape()[2]])

        distort_bbox = distort_bbox[0, 0]
        bboxes = bbox_util.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = bbox_util.bboxes_filter_overlap(labels, bboxes,
                                                         threshold=BBOX_CROP_OVERLAP,
                                                         assign_negative=False)
        if clip_bboxes:
            bboxes = bbox_util.bboxes_clip(distort_bbox, bboxes)
        return cropped_image, labels, bboxes


def preprocess_for_train(image, labels, bboxes, out_height, out_width, bbox,
                         fast_mode, data_format,
                         scope='ssd_preprocessing_train'):
    """Preprocess the given image for training.
    Arguments:
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        used in image distorted crop.
    """
    with tf.name_scope(scope, 'ssd_preprocess_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, channels].')
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        tf_summary_image(image, bboxes, 'image_with_bboxes')

        if data_format == 'NHCW':
            image = tf.transpose(image, perm=(2, 0, 1))
        image = random_rgb_to_gray(image, 0.1)
        dst_image, dst_labels, dst_bboxes = distorted_bounding_box_crop(image,
                                                                        labels,
                                                                        bboxes,
                                                                        bbox,
                                                                        min_object_covered=MIN_OBJECT_COVERED)
        dst_image = tf.image.resize_images(dst_image, [out_height, out_width])
        dst_image, dst_bboxes = random_flip_left_right(dst_image, dst_bboxes)
        dst_image = apply_with_random_selector(dst_image,
                                               lambda x, order: distort_color(x, order, fast_mode),
                                               num_cases=4)
        image = dst_image * 255.
        return image, labels, bboxes


def preprocess_for_eval(image, labels, bboxes,
                        out_height, out_width,
                        data_format,
                        difficults,
                        scope='ssd_preprocessing_eval'):
    """Preprocess an image for evaluation."""
    with tf.name_scope(scope, 'ssd_preprocessing_eval'):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, channels].')

        if data_format == 'NCHW':
            image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.image.resize_images(image, [out_height, out_width],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)

        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)

        return image, labels, bboxes


def  preprocess_image(image,
                     labels,
                     bboxes,
                     out_height=EVAL_SIZE[0], out_width=EVAL_SIZE[1],
                     bbox=None,
                     fast_mode=False,
                     data_format='NHWC',
                     difficults=None,
                     is_training=False):
    if is_training:
        return preprocess_for_train(image, labels, bboxes, out_height, out_width,
                                    bbox, fast_mode, data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes,
                                   out_height, out_width,
                                   data_format, difficults)
