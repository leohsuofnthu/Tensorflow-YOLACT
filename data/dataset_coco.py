import os
import tensorflow as tf

"""
    ref: https://www.tensorflow.org/tutorials/text/image_captioning
"""

# Todo: Download and preprocess coco 2017 train/val

annotation_zip = tf.keras.utils.get_file('annotation.zip',
                                         cache_subdir=os.path.abspath('.'),
                                         origin='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                         extract=True)
annotation_file = os.path.dirname(annotation_zip) + '/annotations/instances_train2017.json'

name_of_zip = 'train2017.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
    image_zip = tf.keras.utils.get_file(name_of_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2017.zip',
                                        extract=True)
    PATH = os.path.dirname(image_zip) + '/train2017/'
else:
    PATH = os.path.abspath('.') + '/train2017/'
