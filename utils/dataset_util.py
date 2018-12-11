"""Contain utilities for convert datasets."""

import os
import sys
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    """Write a file with the list of class names."""
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names.keys():
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Read a file with the list of class names."""
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
