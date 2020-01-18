"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import tensorflow as tf


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)

def get_dataset(tfrecord_dir, subset, batch_size):
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    print(tf.shape(files))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))  # wtf?
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=1024)
    # Todo parser function for mapping and data augmentation
    dataset = dataset.map(map_func=_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    return dataset


d = get_dataset("./coco", "train", 1)
print(d)

count = 1
for sample in d:
    print(sample)
    count -= 1
    if not count:
        break
