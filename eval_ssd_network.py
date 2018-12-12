# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory, ssd_vgg_300
from preprocessing import preprocessing_factory
from utils.bbox_util import bboxes_matching_batch
from utils.metrics import *

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascal2012', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'ssd_vgg_300', 'The name of the preprocessing to use. If left '
                                         'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size_height', None, 'Eval image height in pixel.')

tf.app.flags.DEFINE_integer(
    'eval_image_size_width', None, 'Eval image width in pixel.')

tf.app.flags.DEFINE_integer(
    'select_threshold', 0.01, 'selection threshold.'
)

tf.app.flags.DEFINE_integer(
    'nms_threshold', 0.5, 'Non-Maximum selection threshold.'
)

tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'select top k bboxes per class.'
)

tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Non-Maximum selection keep top k bboxes per class.'
)

tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.'
)

tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'bboxes matching threshold.'
)

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory using fraction.'
)

FLAGS = tf.app.flags.FLAGS

DATA_FORMAT = 'NHWC'


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        ssd_model = ssd_vgg_300.SSDNet()
        network_fn = nets_factory.get_network_fn(
            ssd_model,
            weight_decay=FLAGS.weight_decay,
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size)
        [image, labels, bboxes] = provider.get(['image', 'object/label', 'object/bbox'])
        labels -= FLAGS.labels_offset

        if FLAGS.remove_difficult:
            difficults_gt = provider.get(['object/difficult'])
        else:
            difficults_gt = tf.zeros(tf.shape(labels), dtype=tf.int64)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name)

        eval_image_size_height = FLAGS.eval_image_size_height or ssd_model.ssd_params.image_size[0]
        eval_image_size_width = FLAGS.eval_image_size_width or ssd_model.ssd_params.image_size[1]

        image, labels_gt, bboxes_gt = image_preprocessing_fn(image, labels, bboxes,
                                                             eval_image_size_height, eval_image_size_width,
                                                             data_format=DATA_FORMAT,
                                                             is_training=False)

        anchors = ssd_model.anchors_for_all_layer()
        labels_en, scores_en, bboxes_en = ssd_model.bboxes_encode(anchors, labels_gt, bboxes_gt)

        images, labels_gt, bboxes_gt, difficults_gt, labels_en, scores_en, bboxes_en = \
            tf.train.batch(
            [image, labels_gt, bboxes_gt, difficults_gt,labels_en, scores_en, bboxes_en],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ################################
        # SSD Model + outputs decoding #
        ################################
        logits, locs, endpoints = network_fn(images)
        ssd_model.ssd_class_and_loc_losses(logits, locs, labels_en, bboxes_en, scores_en)

        # Performing post_processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detect objects from SSD Model outputs
            locs_aggr = ssd_model.bboxes_decode(locs, anchors)
            scores_nms, bboxes_nms = ssd_model.detected_bboxes(logits,
                                                               locs_aggr,
                                                               FLAGS.select_threshold,
                                                               FLAGS.nms_threshold,
                                                               FLAGS.select_top_k,
                                                               FLAGS.keep_top_k)

            num_bboxes_gt, tp, fp = bboxes_matching_batch(scores_nms.keys(), scores_nms,
                                                          bboxes_nms, labels_gt, bboxes_gt,
                                                          difficults_gt,
                                                          matching_threshold=FLAGS.matching_threshold)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()


        # Define the metrics:
        with tf.device('/device:CPU:0'):
            dict_metrics = {}
            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                # op = tf.Print(op, [metric[0]], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # FP and TP metrics.
            tp_fp_metric = streaming_tp_fp_arrays(num_bboxes_gt, tp, fp, scores_nms)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                tp_fp_metric[1][c])

            # Add to summaries precision/recall values.
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = precision_recall(*tp_fp_metric[0][c])

                # Average precision VOC12.
                v = average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v

            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore)


if __name__ == '__main__':
    tf.app.run()
