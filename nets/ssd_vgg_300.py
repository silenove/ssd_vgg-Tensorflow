"""300 VGG-based SSD model.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

@@ssd_vgg_300
"""

import numpy as np
import tensorflow as tf

import math
from collections import namedtuple
from utils import network_util, bbox_util

slim = tf.contrib.slim

SSD_params = namedtuple('SSD_params', ['image_size', 'batch_size', 'num_classes', 'featmap_layers',
                                       'featmap_size', 'num_anchors', 'anchor_steps', 'anchor_offset',
                                       'S_min', 'S_max', 'box_scales', 'anchor_ratios',
                                       'prior_variance'])

class SSDNet(object):

    def __init__(self, ssd_params=None):
        if ssd_params is None:
            self.ssd_params = SSD_params(image_size=(300, 300),
                                         batch_size=4,
                                         num_classes=21,
                                         featmap_layers=['conv4', 'conv7', 'conv8',
                                                         'conv9', 'conv10', 'conv11'],
                                         featmap_size=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)], # (h, w)
                                         num_anchors=[4, 6, 6, 6, 4, 4],
                                         anchor_steps=[8, 16, 32, 64, 100, 300],
                                         anchor_offset=0.5,
                                         S_min=0.15,
                                         S_max=0.9,
                                         box_scales=[],
                                         anchor_ratios=[[2, 1 / 2],
                                                        [2, 3, 1 / 2, 1 / 3],
                                                        [2, 3, 1 / 2, 1 / 3],
                                                        [2, 3, 1 / 2, 1 / 3],
                                                        [2, 1 / 2],
                                                        [2, 1 / 2]],
                                         prior_variance=[0.1, 0.1, 0.2, 0.2])
        else:
            self.ssd_params = ssd_params

        self._compute_box_scales()

    def set_batch_size(self, batch_size):
        self.ssd_params = self.ssd_params._replace(batch_size=batch_size)

    def _ssd_vgg_300_base_network(self, inputs, reuse=None, scope=None):

        """Define the base nets of 300 VGG-based SSD.
        input image : batch_size x 300 x 300 x channels.
        convolution layers default stride = 1, padding = 'SAME'
        maxpool layers default stride = 2, padding = 'VALID'

        """
        end_points = {}

        with tf.variable_scope(scope, 'ssd_vgg_300', [inputs], reuse=reuse):
            # Original VGG-16 nets
            # input: batch_size x 300 x 300 x channels
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            end_points['conv1'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

            # tensor: batch_size x 150 x 150 x 64
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            end_points['conv2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

            # tensor: batch_size x 75 x 75 x 128
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            end_points['conv3'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')

            # tensor: batch_size x 38 x 38 x 256
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            end_points['conv4'] = net
            net = slim.max_pool2d(net, [2, 2], stride=1, padding='SAME', scope='pool4')

            # tensor: batch_size x 38 x 38 x 512
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            end_points['conv5'] = net
            net = slim.max_pool2d(net, [2, 2], stride=1, padding='SAME', scope='pool5')

            # SSD nets
            # tensor: batch_size x 38 x 38 x 512
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['conv6'] = net

            # tensor: batch_size x 19 x 19 x 1024
            net = slim.conv2d(net, 1024, [1, 1], stride=2, scope='conv7')
            end_points['conv7'] = net
            net = slim.max_pool2d(net, [2, 2], stride=1, padding='SAME', scope='pool7')

            # tensor: batch_size 19 x 19 x 1024
            net = slim.conv2d(net, 256, [1, 1], scope='conv8_1x1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv8_3x3')
            end_points['conv8'] = net

            # tensor: batch_size x 10 x 10 x 512
            net = slim.conv2d(net, 128, [1, 1], scope='conv9_1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_3x3')
            end_points['conv9'] = net

            # tensor: batch_size x 5 x 5 x 256
            net = slim.conv2d(net, 128, [1, 1], scope='conv10_1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv10_3x3')
            end_points['conv10'] = net

            # tensor: batch_size x 3 x 3 x 256
            net = slim.conv2d(net, 128, [1, 1], scope='conv11_1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=1, padding='VALID', scope='conv11_3x3')
            end_points['conv11'] = net
            # tensor: batch_size x 1 x 1 x 256

            for i, layer in enumerate(self.ssd_params.featmap_layers):
                if i == 0:
                    # conv4 layer
                    with tf.variable_scope(layer+'_mbox'):
                        # classes
                        norm = slim.batch_norm(end_points[layer], decay=0.9997, epsilon=0.000001,
                                                       scope=layer + '_norm')
                        norm_mbox_conf_perm = slim.conv2d(norm,
                                                          self.ssd_params.num_classes * self.ssd_params.num_anchors[i],
                                                          [3, 3],
                                                          scope=layer + '_norm_mbox_conf_perm')
                        norm_mbox_conf_flat = tf.contrib.layers.flatten(norm_mbox_conf_perm,
                                                                        scope=layer + '_norm_mbox_conf_flat')
                        end_points[layer + '_mbox_conf_flat'] = norm_mbox_conf_flat

                        # bounding box
                        norm_mbox_loc_perm = slim.conv2d(norm,
                                                         self.ssd_params.num_anchors[i] * 4,
                                                         [3, 3],
                                                         scope=layer + '_norm_mbox_loc_perm')
                        norm_mbox_loc_flat = tf.contrib.layers.flatten(norm_mbox_loc_perm,
                                                                       scope=layer + '_norm_mbox_loc_flat')
                        end_points[layer + '_mbox_loc_flat'] = norm_mbox_loc_flat
                else:
                    # conv7, conv8, conv9, conv10, conv11
                    with tf.variable_scope(layer+'_mbox'):
                        # classes
                        mbox_conf_perm = slim.conv2d(end_points[layer],
                                                     self.ssd_params.num_classes * self.ssd_params.num_anchors[i],
                                                     [3, 3],
                                                     scope=layer + '_mbox_conf_perm')
                        mbox_conf_flat = tf.contrib.layers.flatten(mbox_conf_perm,
                                                                   scope=layer + '_mbox_conf_flat')
                        end_points[layer + '_mbox_conf_flat'] = mbox_conf_flat

                        # bounding box
                        mbox_loc_perm = slim.conv2d(end_points[layer],
                                                    self.ssd_params.num_anchors[i] * 4,
                                                    [3, 3],
                                                    scope=layer + '_mbox_loc_perm')
                        mbox_loc_flat = tf.contrib.layers.flatten(mbox_loc_perm,
                                                                  scope=layer + '_mbox_loc_flat')
                        end_points[layer + '_mbox_loc_flat'] = mbox_loc_flat

            # concatenate and reshape
            mbox_conf = tf.concat([end_points[layer + '_mbox_conf_flat'] for layer in self.ssd_params.featmap_layers],
                                  axis=-1)
            mbox_conf_reshape = tf.reshape(mbox_conf, [self.ssd_params.batch_size, -1, self.ssd_params.num_classes])
            end_points['mbox_conf_reshape'] = mbox_conf_reshape

            mbox_loc = tf.concat([end_points[layer + '_mbox_loc_flat'] for layer in self.ssd_params.featmap_layers],
                                 axis=-1)
            mbox_loc_reshape = tf.reshape(mbox_loc, [self.ssd_params.batch_size, -1, 4])
            end_points['mbox_loc_reshape'] = mbox_loc_reshape

            return end_points['mbox_conf_reshape'], end_points['mbox_loc_reshape'], end_points

    def ssd_vgg_300_net(self, inputs, is_training=True, reuse=None, scope='ssd_vgg_300'):
        """Creates the 300 VGG-based SSD model."""
        end_points = {}
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            logits, locs, end_points = self._ssd_vgg_300_base_network(inputs, reuse, scope=scope)

            return logits, locs, end_points


    def ssd_arg_scope(self,
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer(),
                      weight_decay=0.00004):
        """Define ssd arg scope."""
        with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                            weights_initializer=weights_initializer,
                            biases_initializer=biases_initializer,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _compute_box_scales(self):
        """Compute the scales of the default boxes for each feature map."""
        num_layers = len(self.ssd_params.featmap_layers)
        for k in range(num_layers):
            if k == 0:
                self.ssd_params.box_scales.append(self.ssd_params.S_min)
            elif k == num_layers - 1:
                self.ssd_params.box_scales.append(self.ssd_params.S_max)
            else:
                s_k = self.ssd_params.S_min + (self.ssd_params.S_max - self.ssd_params.S_min) / \
                      (num_layers - 1) * (k - 1)
                self.ssd_params.box_scales.append(s_k)

    def _anchor_for_one_layer(self, layer_idx, dtype=np.float32):
        """Compute the relative coordinate of the SSD default anchors for a one feature layer.

        Arguments:
            layer_idx: the index of feature layer.

        Return:
            y, x, w, h: the anchors coordinate.
        """
        y, x = np.mgrid[0:self.ssd_params.featmap_size[layer_idx][0],
               0:self.ssd_params.featmap_size[layer_idx][1]]
        y = y.astype(dtype)
        x = x.astype(dtype)
        y = ((y + self.ssd_params.anchor_offset) * self.ssd_params.anchor_steps[layer_idx]) / \
            self.ssd_params.image_size[0]
        x = ((x + self.ssd_params.anchor_offset) * self.ssd_params.anchor_steps[layer_idx]) / \
            self.ssd_params.image_size[1]

        # Change the shape to h x w x num_anchors.
        y = np.expand_dims(y, axis=-1)
        y = np.concatenate([y for _ in range(self.ssd_params.num_anchors[layer_idx])], axis=-1)
        x = np.expand_dims(x, axis=-1)
        x = np.concatenate([x for _ in range(self.ssd_params.num_anchors[layer_idx])], axis=-1)

        h = np.zeros([self.ssd_params.num_anchors[layer_idx]], dtype=dtype)
        w = np.zeros([self.ssd_params.num_anchors[layer_idx]], dtype=dtype)

        for i in range(self.ssd_params.num_anchors[layer_idx]):
            if i == 0:
                h[i] = self.ssd_params.box_scales[layer_idx]
                w[i] = self.ssd_params.box_scales[layer_idx]
            elif i == self.ssd_params.num_anchors[layer_idx] - 1:
                if layer_idx < len(self.ssd_params.featmap_layers) - 1:
                    s = np.sqrt(self.ssd_params.box_scales[layer_idx] *
                                self.ssd_params.box_scales[layer_idx + 1])
                    h[i] = s
                    w[i] = s
                else:
                    # The last feature map.
                    s = np.sqrt(self.ssd_params.box_scales[layer_idx] *
                                (self.ssd_params.box_scales[layer_idx] + 1) / 2)
                    h[i] = s
                    w[i] = s
            else:
                for a_r in self.ssd_params.anchor_ratios:
                    h[i] = self.ssd_params.box_scales[layer_idx] * \
                           np.sqrt(self.ssd_params.anchor_ratios[i - 1][0])
                    w[i] = self.ssd_params.box_scales[layer_idx] / \
                           np.sqrt(self.ssd_params.anchor_ratios[i - 1][1])

        return y, x, h, w

    def anchors_for_all_layer(self):
        """Compute the relative coordinate of the SSD default anchors for all feature map."""
        anchors_cords = {}
        for i, layer in enumerate(self.ssd_params.featmap_layers):
            anchors_cords[layer] = self._anchor_for_one_layer(i)
        return anchors_cords

    def bboxes_encode(self, anchors, gt_labels, gt_bboxes,
                      match_threshold=0.5, dtype=tf.float32, scope=None):
        """Encode labels and bounding boxes."""
        labels, scores, locs = network_util.encode_gtbboxes_from_all_layers(self.ssd_params,
                                                                            anchors,
                                                                            gt_labels,
                                                                            gt_bboxes,
                                                                            match_threshold,
                                                                            dtype,
                                                                            scope)
        labels = tf.concat([tf.reshape(x, [-1]) for x in labels], axis=0)
        scores = tf.concat([tf.reshape(x, [-1]) for x in scores], axis=0)
        locs = tf.concat([tf.reshape(x, [-1]) for x in locs], axis=0)
        return labels, scores, locs

    def bboxes_decode(self, locs_pred, anchors, scope='ssd_bboxes_decode'):
        """Decode labels and bounding boxes."""
        bboxes = []
        for i, layer in enumerate(self.ssd_params.featmap_layers):
            y, x, h, w = anchors[layer]
            featmap_h = self.ssd_params.featmap_size[i][0]
            featmap_w = self.ssd_params.featmap_size[i][1]
            h = tf.reshape(tf.concat([h]*featmap_h*featmap_w, axis=-1),
                           [featmap_h, featmap_w, -1])
            w = tf.reshape(tf.concat([w]*featmap_h*featmap_w, axis=-1),
                           [featmap_h, featmap_w, -1])
            bboxes.append(tf.reshape(tf.stack([y,x,h,w], axis=-1), [-1, 4]))
        bboxes = tf.concat(bboxes, axis=0)
        bboxes_batches = tf.concat([tf.expand_dims(bboxes, axis=0)]*self.ssd_params.batch_size, axis=0)
        anchors = [bboxes_batches[:, :, 0], bboxes_batches[:, :, 1],
                   bboxes_batches[:, :, 2], bboxes_batches[:, :, 3]]
        return network_util.decode_bboxes_from_all_layer(locs_pred,
                                                         anchors,
                                                         prior_scaling=self.ssd_params.prior_variance,
                                                         scope=scope)

    def detected_bboxes(self, predictions, localizations, select_threshold=0.01,
                        nms_threshold=0.5, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from SSD Model output."""
        scores_select, bboxes_select = network_util.select_detected_bboxes_all_classes(predictions,
                                                                           localizations,
                                                                           self.ssd_params.num_classes,
                                                                           select_threshold)
        dict_scores_sorted, dict_bboxes_sorted = bbox_util.bboxes_sort_all_classes(scores_select,
                                                                                   bboxes_select,
                                                                                   top_k)
        dict_scores_nms, dict_bboxes_nms = bbox_util.bboxes_nms_all_classes(dict_scores_sorted,
                                                                           dict_bboxes_sorted,
                                                                           self.ssd_params.batch_size,
                                                                           nms_threshold,
                                                                           keep_top_k)
        return dict_scores_nms, dict_bboxes_nms

    def ssd_class_and_loc_losses(self,
                                 logits_pred,
                                 localization_pred,
                                 classes_gt,
                                 localization_gt,
                                 scores_gt,
                                 match_threshold=0.5,
                                 negative_ratio=3.0,
                                 alpha=1.0,
                                 label_smoothing=0.0,
                                 scope=None):
        """Compute the SSD nets losses including classification and localization.

        Arguments:
            logits_pred: SSD nets output, batch_size x -1 x 21.
            localization_pred: SSD nets output, batch_size x -1 x 4.
            classes_gt: gt classes,
        """
        with tf.name_scope(scope, 'ssd_losses'):
            # Reshape all tensors.
            logits_pred_flat = tf.reshape(logits_pred, [-1, self.ssd_params.num_classes])
            localization_pred_flat = tf.reshape(localization_pred, [-1, 4])
            classes_gt_flat = tf.reshape(classes_gt, [-1])
            localization_gt_flat = tf.reshape(localization_gt, [-1, 4])
            scores_gt_flat = tf.reshape(scores_gt, [-1])
            dtype = logits_pred_flat.dtype

            # Compute positive matching mask
            posi_mask = scores_gt_flat > match_threshold
            f_posi_mask = tf.cast(posi_mask, dtype)
            n_positive = tf.reduce_sum(f_posi_mask)

            # Hard negative mining
            neg_mask = tf.logical_not(posi_mask)
            f_neg_mask = tf.cast(neg_mask, dtype)
            logits_pred_softmax = slim.softmax(logits_pred_flat)
            neg_values = tf.where(neg_mask, logits_pred_softmax[:, 0], 1. - f_neg_mask)

            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(f_neg_mask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positive, tf.int32)
            n_neg = tf.minimum(n_neg, max_neg_entries)

            vals, idxes = tf.nn.top_k(-1 * neg_values, k=n_neg)
            max_hard_pred = -1 * vals[-1]
            # Final negative mask
            neg_mask = tf.logical_and(neg_mask, neg_values < max_hard_pred)
            f_neg_mask = tf.cast(neg_mask, dtype)

            with tf.name_scope('cross_entropy'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred_flat,
                                                                      labels=classes_gt_flat)
                loss_posi = tf.div(tf.reduce_sum(loss * f_posi_mask), tf.reduce_sum(f_posi_mask),
                                   name='positive_loss')
                loss_neg = tf.div(tf.reduce_sum(loss * f_neg_mask), tf.reduce_sum(f_neg_mask),
                                  name='negative_loss')
                tf.losses.add_loss(loss_posi)
                tf.losses.add_loss(loss_neg)
            with tf.name_scope('localization_loss'):
                loss = network_util.abs_smooth_L1(localization_pred_flat - localization_gt_flat)
                loss_loc = tf.div(tf.reduce_sum(loss * tf.expand_dims(f_posi_mask, axis=-1)) * alpha,
                                  tf.reduce_sum(f_posi_mask),
                                  name='localization_loss')
                tf.losses.add_loss(loss_loc)






