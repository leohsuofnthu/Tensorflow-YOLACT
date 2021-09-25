"""
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py
"""

import hashlib
import io
import json
import os

import PIL.Image
import contextlib2
import numpy as np

from absl import app
from absl import flags
from absl import logging
from pycocotools import mask

from data.coco_tfrecord_utils import *

FLAGS = flags.FLAGS

flags.DEFINE_boolean('include_masks', True,
                     'Whether to include instance segmentations masks (PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_image_dir', 'D:/project5-YOLACT/Tensorflow-YOLACT/data/train2017',
                    'Training image directory.')
flags.DEFINE_string('val_image_dir', 'D:/project5-YOLACT/Tensorflow-YOLACT/data/val2017',
                    'Validation image directory.')
flags.DEFINE_string('test_image_dir', '',
                    'Test image directory.')
flags.DEFINE_string('train_annotations_file',
                    'D:/project5-YOLACT/Tensorflow-YOLACT/data/annotations/instances_train2017.json',
                    'Training annotations JSON file.')
flags.DEFINE_string('val_annotations_file',
                    'D:/project5-YOLACT/Tensorflow-YOLACT/data/annotations/instances_val2017.json',
                    'Validation annotations JSON file.')
flags.DEFINE_string('testdev_annotations_file', '',
                    'Test-dev annotations JSON file.')
flags.DEFINE_string('output_dir', './coco', 'Output data directory.')

logging.set_verbosity(logging.INFO)


def create_tf_example(image, annotations_list, image_dir, category_index, include_masks=True):
    """Converts image and annotations to a tf.Example proto."""

    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin, xmax, ymin, ymax = [], [], [], []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0

    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        # ignore wrongly annotated data
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue

        # normalized bbox
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)

        # other attributes
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        # category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
            try:
                # if the seg annotation is not RLE, need to be convert
                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
            except:
                # if the seg annotation is already RLE
                binary_mask = mask.decode(object_annotations['segmentation'])

            if not object_annotations['iscrowd']:
                if binary_mask.ndim == 3:  # for COCO dataset
                    binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
        'image/height':
            int64_feature(image_height),
        'image/width':
            int64_feature(image_width),
        'image/filename':
            bytes_feature(filename.encode('utf8')),
        'image/source_id':
            bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            bytes_feature(key.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_jpg),
        'image/format':
            bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            float_list_feature(xmin),
        'image/object/bbox/xmax':
            float_list_feature(xmax),
        'image/object/bbox/ymin':
            float_list_feature(ymin),
        'image/object/bbox/ymax':
            float_list_feature(ymax),
        # 'image/object/class/text':
        #     bytes_list_feature(category_names),
        'image/object/class/label_id':
            int64_list_feature(category_ids),
        'image/object/is_crowd':
            int64_list_feature(is_crowd),
        'image/object/area':
            float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (bytes_list_feature(encoded_mask_png))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, include_masks, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format."""

    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.io.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = create_category_index(groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        logging.info('%d images are missing annotations.',
                     missing_annotation_count)

        total_num_annotations_skipped = 0
        for idx, image in enumerate(images):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            if len(annotations_list) > 0:
                _, tf_example, num_annotations_skipped = create_tf_example(
                    image, annotations_list, image_dir, category_index, include_masks)
                total_num_annotations_skipped += num_annotations_skipped
                shard_idx = idx % num_shards
                if tf_example:
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        logging.info('Finished writing, skipped %d annotations.',
                     total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    # testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

    _create_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=100)

    _create_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        num_shards=50)
    """
    _create_tf_record_from_coco_annotations(
        FLAGS.testdev_annotations_file,
        FLAGS.test_image_dir,
        testdev_output_path,
        FLAGS.include_masks,
        num_shards=100)
    """


if __name__ == '__main__':
    app.run(main)
