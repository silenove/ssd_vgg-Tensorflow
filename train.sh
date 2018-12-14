#! /bin/bash

DATASET_DIR=./tfrecords
TRAIN_DIR=./log/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt

python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascal2012\
    --dataset_split_name=train \
    --model_name=ssd_vgg_300 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_vgg_300/conv6,ssd_vgg_300/conv7,ssd_vgg_300/conv8_1x1,ssd_vgg_300/conv8_3x3,ssd_vgg_300/conv9_1x1,ssd_vgg_300/conv9_3x3,ssd_vgg_300/conv10_1x1,ssd_vgg_300/conv10_3x3,ssd_vgg_300/conv11_1x1,ssd_vgg_300/conv11_3x3,ssd_vgg_300/conv4_mbox,ssd_vgg_300/conv7_mbox,ssd_vgg_300/conv8_mbox,ssd_vgg_300/conv9_mbox,ssd_vgg_300/conv10_mbox,ssd_vgg_300/conv11_mbox \
    --trainable_scopes=ssd_vgg_300/conv6,ssd_vgg_300/conv7,ssd_vgg_300/conv8_1x1,ssd_vgg_300/conv8_3x3,ssd_vgg_300/conv9_1x1,ssd_vgg_300/conv9_3x3,ssd_vgg_300/conv10_1x1,ssd_vgg_300/conv10_3x3,ssd_vgg_300/conv11_1x1,ssd_vgg_300/conv11_3x3,ssd_vgg_300/conv4_mbox,ssd_vgg_300/conv7_mbox,ssd_vgg_300/conv8_mbox,ssd_vgg_300/conv9_mbox,ssd_vgg_300/conv10_mbox,ssd_vgg_300/conv11_mbox \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=8 \
    --max_number_of_steps=10000
