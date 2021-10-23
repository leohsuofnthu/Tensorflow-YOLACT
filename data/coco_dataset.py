"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf

from data import coco_tfrecord_parser


class ObjectDetectionDataset:

    def __init__(self, dataset_name, tfrecord_dir, anchor_instance, **parser_params):
        self.dataset_name = dataset_name
        self.tfrecord_dir = tfrecord_dir
        self.anchor_instance = anchor_instance
        self.parser_params = parser_params

    def get_dataloader(self, subset, batch_size):
        # function for per-element transformation
        parser = coco_tfrecord_parser.Parser(anchor_instance=self.anchor_instance,
                                             mode=subset,
                                             **self.parser_params)
        # get tfrecord file names
        filenames = tf.io.matching_files(os.path.join(self.tfrecord_dir, f"{subset}.*"))

        # ignore reading order
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed

        # apply suffle and repeat only on traininig data
        if subset == 'train':
            # automatically interleaves reads from multiple files
            dataset = tf.data.TFRecordDataset(filenames)
            # uses data as soon as it streams in, rather than in its original order
            dataset = dataset.with_options(ignore_order)
            dataset = dataset.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        elif subset == 'val' or 'test':
            dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files

        else:
            raise ValueError('Illegal subset name.')

        # apply per-element transformation
        dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
