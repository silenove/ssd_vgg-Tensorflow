"""Convert raw PASCAL VOC dataset to TFRecord for object detection.

The Example proto contains the following fields:
    image/encoded: string, containing JPEG image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/format: string, specifying the format, like 'JPEG'

    image/object/bbox/xmin: list of float specifying the bboxes.
    image/object/bbox/xmax: list of float specifying the bboxes.
    image/object/bbox/ymin: list of float specifying the bboxes.
    image/object/bbox/ymax: list of float specifying the bboxes.
    image/object/bbox/label: list of integer specifying the classification index.
"""

import os
import sys
import random

import numpy as np
import tensorflow as tf

from xml.etree import ElementTree
from utils.dataset_util import write_label_file
from datasets.pascal2012 import VOC_LABELS, SPLIT_TO_SIZES
from utils.dataset_util import int64_list_feature, bytes_list_feature, float_list_feature, \
    bytes_feature, int64_feature

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# The number of images(total 17125) in the validation set.
_NUM_VALIDATION = SPLIT_TO_SIZES['validation']

# Seed for repeatability
_RANDOM_SEED = 123

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', '', 'The dataset directory where the dataset is stored.')
tf.app.flags.DEFINE_string('meta_directory', '', 'The directory containing images and annotations dir')

def _image_to_tfexample(image_name, annotation_name):
    """Generate a tf example by image and annotation file."""
    image_data = tf.gfile.FastGFile(image_name, 'rb').read()
    tree = ElementTree.parse(annotation_name)
    root = tree.getroot()

    # image shape
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    channels = int(size.find('depth').text)

    # image annotations
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label_name = obj.find('name').text
        labels.append(int(VOC_LABELS[label_name][0]))
        labels_text.append(label_name.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        xmin.append(float(bbox.find('xmin').text) / width)
        xmax.append(float(bbox.find('xmax').text) / width)
        ymin.append(float(bbox.find('ymin').text) / height)
        ymax.append(float(bbox.find('ymax').text) / height)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(b'JPEG'),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(channels),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/bbox/label': int64_list_feature(labels),
        'image/object/bbox/text': bytes_list_feature(labels_text),
        'image/object/bbox/difficult': int64_list_feature(difficult),
        'image/object/bbox/truncated': int64_list_feature(truncated),
    }))
    return example

def _get_dataset_name(dataset_dir, split_name):
    output_filename = 'data_%s.tfrecord' % split_name
    return os.path.join(dataset_dir, output_filename)

def _dataset_exist(dataset_dir):
    for split_name in ['train', 'validation']:
        output_filename = _get_dataset_name(dataset_dir, split_name)
        if not tf.gfile.Exists(output_filename):
            return False
    return True

def _get_filenames(dataset_dir):
    meta_names = []
    image_dir = os.path.join(dataset_dir, DIRECTORY_IMAGES)
    for filename in os.listdir(image_dir):
        meta_names.append(filename[:-4])
    return meta_names

def _convert_dataset(split_name, filenames, dataset_dir, meta_dir):
    """Convert the given filenames to a TFRecord dataset."""
    assert split_name in ['train', 'validation']

    output_filename = _get_dataset_name(dataset_dir, split_name)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(filenames)):
            sys.stdout.write('\r>> Converting image %d/%d to %s dataset.' % (i+1, len(filenames), split_name))
            sys.stdout.flush()

            imagename = os.path.join(meta_dir, DIRECTORY_IMAGES, filenames[i] + '.jpg')
            anotname = os.path.join(meta_dir, DIRECTORY_ANNOTATIONS, filenames[i] + '.xml')
            if tf.gfile.Exists(imagename) and tf.gfile.Exists(anotname):
                example = _image_to_tfexample(imagename, anotname)
                tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(_):
    """Run the conversion operation."""
    if not tf.gfile.Exists(FLAGS.dataset_dir):
        tf.gfile.MakeDirs(FLAGS.dataset_dir)

    if _dataset_exist(FLAGS.dataset_dir):
        print('Dataset files already exist. Existing without recreate files.')
        return

    classes_id_to_name = {value[0]: key for key, value in VOC_LABELS.items()}
    meta_filenames = _get_filenames(FLAGS.meta_directory)

    # Divide into training and validation
    random.seed()
    random.shuffle(meta_filenames)

    train_filenames = meta_filenames[_NUM_VALIDATION:]
    validation_filenames = meta_filenames[:_NUM_VALIDATION]

    # convert the training and validation
    _convert_dataset('train', train_filenames, FLAGS.dataset_dir, FLAGS.meta_directory)
    _convert_dataset('validation', validation_filenames, FLAGS.dataset_dir, FLAGS.meta_directory)

    # write the labels file
    write_label_file(classes_id_to_name, FLAGS.dataset_dir)

    print('\nFinished converting the PASCAL VOC dataset.')

if __name__ == '__main__':
    tf.app.run()
