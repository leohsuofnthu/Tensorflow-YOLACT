"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import tensorflow as tf


def decode_png_mask(image_buffer):
    image = tf.squeeze(
        tf.image.decode_png(image_buffer, channels=1), axis=2)
    image.set_shape([None, None])
    image = tf.cast(tf.greater(image, 0), dtype=tf.float32)
    return image


# the features we want to extract
def _parse_fn(example_serialized):
    feature_map = {
        'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
        'image/object/is_crowd': tf.io.VarLenFeature(dtype=tf.int64),
        'image/object/mask': tf.io.VarLenFeature(dtype=tf.string),
    }
    # need to be parsed in a batch way
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    height = parsed['image/height']
    width = parsed['image/width']
    img = tf.image.decode_jpeg(parsed['image/encoded'])
    img = tf.image.resize(img, [550, 550])
    png_masks = parsed['image/object/mask']
    png_masks = tf.sparse.to_dense(png_masks, default_value='')
    masks = tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.cast(tf.stack([0, height, width]), dtype=tf.int32)))
    iscrowd = tf.sparse.to_dense(parsed['image/object/is_crowd'])
    labels = tf.sparse.to_dense(parsed['image/object/class/text'])
    xmin = tf.sparse.to_dense(parsed['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed['image/object/bbox/ymax'])

    tensor_dict = dict([('image', img), ('masks', masks), ('xmin', xmin), ('xmax', xmax), ('ymin', ymin), ('ymax', ymax),
                       ('iscrowd', iscrowd)])

    tensor_dict['image'].set_shape([None, None, 3])
    tensor_dict['masks'].set_shape([None, None, None])
    tensor_dict['xmin'].set_shape([None])
    tensor_dict['ymin'].set_shape([None])
    tensor_dict['xmax'].set_shape([None])
    tensor_dict['ymax'].set_shape([None])
    tensor_dict['iscrowd'].set_shape([None])

    tf.print("img", tf.shape(img))
    tf.print("masks", tf.shape(masks))
    tf.print("label", labels)
    tf.print("xmin", xmin)
    tf.print("xmax", xmax)
    tf.print("isCrowd", iscrowd)
    tf.print("ymin", ymin)
    tf.print("ymax", ymax)
    return tensor_dict


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
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    return dataset


d = get_dataset("./coco", "train", 2)
print(d)

count = 1
for sample in d:
    print(sample)
    count -= 1
    if not count:
        break
