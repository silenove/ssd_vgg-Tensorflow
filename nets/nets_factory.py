# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import ssd_vgg_300

slim = tf.contrib.slim


def get_network_fn(model, weight_decay=0.00004, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the nets.
      num_classes: The number of classes to use for classification. If 0 or None,
        the logits layer is omitted and its input features are returned instead.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
            net, end_points = network_fn(images)
        The `images` input is a tensor of shape [batch_size, height, width, 3]
        with height = width = network_fn.default_image_size. (The permissibility
        and treatment of other sizes depends on the network_fn.)
        The returned `end_points` are a dictionary of intermediate activations.
        The returned `net` is the topmost layer, depending on `num_classes`:
        If `num_classes` was a non-zero integer, `net` is a logits tensor
        of shape [batch_size, num_classes].
        If `num_classes` was 0 or `None`, `net` is a tensor with the input
        to the logits layer of shape [batch_size, 1, 1, num_features] or
        [batch_size, num_features]. Dropout has not been applied to this
        (even if the nets's original classification does); it remains for
        the caller to do this or not.

    Raises:
      ValueError: If nets `name` is not recognized.
    """
    func = model.ssd_vgg_300_net

    @functools.wraps(func)
    def network_fn(inputs, **kwargs):
        arg_scope = model.ssd_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(inputs, is_training=is_training, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
