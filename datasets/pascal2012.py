"""Provide the Dataset of Pascal Voc dataset."""

import os
import tensorflow as tf
from utils.dataset_util import read_label_file

slim = tf.contrib.slim

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

TOTAL_SIZE = 17125

SPLIT_TO_SIZES = {'train': 16125, 'validation': 1000}

NUM_CLASSES = 21  # include background

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'image/height': 'The height of the image in pixel.',
    'image/width': 'The width of the image in pixel.',
    'image/channels': 'The channels of the image.',
    'object/bbox': 'The bboxes(ymin, xmin, ymax, xmax) of all objects in the image.',
    'object/label': 'The labels of all objects in the image.',
    'object/difficult': 'tf.int64, the difficulties to recognize every object in the image.',
    'object/truncated': 'tf.int64, the truncation of all objects in the image.',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Get a dataset with instructions for reading PASCAL VOC dataset."""
    if split_name not in ['train', 'validation']:
        raise ValueError('split name %s is not recognized.' % split_name)
    if not file_pattern:
        file_pattern = 'data_%s.tfrecord'
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'image/height': slim.tfexample_decoder.Tensor('image/height'),
        'image/width': slim.tfexample_decoder.Tensor('image/width'),
        'image/channels': slim.tfexample_decoder.Tensor('image/channels'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox'
        ),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLIT_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names,
    )
