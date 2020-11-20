"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf

from data import anchor
from data import coco_tfrecord_parser

import config as cfg


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)
def prepare_dataloader(tfrecord_dir, batch_size, subset="train", **parser_params):
    anchorobj = anchor.Anchor(**cfg.anchor_params)
    parser = coco_tfrecord_parser.Parser(anchor_instance=anchorobj,
                                         mode=subset,
                                         **parser_params)

    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    num_shards = tf.cast(tf.shape(files)[0], tf.int64)
    shards = tf.data.Dataset.from_tensor_slices(files)

    if subset == 'train':
        shards = shards.shuffle(num_shards)
        shards = shards.repeat()
        dataset = shards.interleave(tf.data.TFRecordDataset,
                                    cycle_length=num_shards,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=2048)
    else:
        dataset = tf.data.TFRecordDataset(shards)

    dataset = dataset.map(map_func=parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
