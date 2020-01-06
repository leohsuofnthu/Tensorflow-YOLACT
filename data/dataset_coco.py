import os
import json
import tensorflow as tf

"""
    Pre-Processing COCO 2017 from scratch.
    Download => process label and image file path => Pre-process and img load function => make tf dataset 
    ref: https://www.tensorflow.org/tutorials/text/image_captioning
"""

# Todo: Download and pre-process coco 2017 train/val

# Download annotation file (annotation contains train/val)
annotation_zip = tf.keras.utils.get_file("annotation.zip",
                                         cache_subdir=os.path.abspath('.'),
                                         origin="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                         extract=True)

annotation_train = os.path.dirname(annotation_zip) + "/annotations/instances_train2017.json"
annotation_valid = os.path.dirname(annotation_zip) + "/annotations/instances_val2017.json"

train_zip = "train2017.zip"
val_zip = "val2017.zip"

# Download train images
if not os.path.exists(os.path.abspath('.') + '/' + train_zip):
    image_zip = tf.keras.utils.get_file(train_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2017.zip',
                                        extract=True)
    PATH_train = os.path.dirname(image_zip) + '/train2017/'
else:
    PATH_train = os.path.abspath('.') + '/train2017/'

# Download validation images
if not os.path.exists(os.path.abspath('.') + '/' + val_zip):
    image_zip = tf.keras.utils.get_file(val_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/val2017.zip',
                                        extract=True)
    PATH_valid = os.path.dirname(image_zip) + '/val2017/'
else:
    PATH_valid = os.path.abspath('.') + '/val2017/'

# Read the json file
print("Load annotation .json")
with open(annotation_train, 'r') as f:
    annotations = json.load(f)
