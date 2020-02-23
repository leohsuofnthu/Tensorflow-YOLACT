"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf

from data import anchor
from data import yolact_parser


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)
def prepare_dataloader(tfrecord_dir, batch_size, subset="train"):
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    # print(tf.shape(files))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))  # wtf?
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=1024)
    anchorobj = anchor.Anchor(img_size=550,
                              feature_map_size=[69, 35, 18, 9, 5],
                              aspect_ratio=[1, 0.5, 2],
                              scale=[24, 48, 96, 192, 384])

    parser = yolact_parser.Parser(output_size=550,
                                  anchor_instance=anchorobj,
                                  match_threshold=0.5,
                                  unmatched_threshold=0.5,
                                  mode=subset)

    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
