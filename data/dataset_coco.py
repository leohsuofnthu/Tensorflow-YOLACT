"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import tensorflow as tf


# the features we want to extract
def _parse_fn(example_serialized):
    feature_map = {
        'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], dtype=tf.float32),
        'image/object/is_crowd': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/object/mask': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    # need to be parsed in a batch way
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    img = tf.image.decode_image(parsed['image/encoded'])
    """
    mask = tf.image.decode_image(parsed['image/object/mask'])
    iscrowd = parsed['image/object/is_crowd']
    xmin = parsed['image/object/bbox/xmin']
    xmax = parsed['image/object/bbox/xmax']
    ymin = parsed['image/object/bbox/ymin']
    ymax = parsed['image/object/bbox/ymax']
    return img, mask, iscrowd, xmin, xmax, ymin, ymax
    """
    return img


# Todo encapsulate it as a class

def get_dataset(tfrecord_dir, subset, batch_size):
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    print(tf.shape(files))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))  # wtf?
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=1024)
    # Todo parser function for mapping and data augmentation
    dataset = dataset.map(map_func=_parse_fn, num_parallel_calls=4)
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(batch_size)

    return dataset


d = get_dataset("./coco", "train", 2)
print(d)


count = 1
for sample in d:
    print(tf.shape(sample))
    count -= 1
    if not count:
        break

