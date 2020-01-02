import tensorflow as tf
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
dataset = tfds.load(name="coco2017", split=tfds.Split.TRAIN)

dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for features in dataset.take(1):
    print(features)
