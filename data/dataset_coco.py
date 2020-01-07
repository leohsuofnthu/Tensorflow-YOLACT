import tensorflow_datasets as tfds
tfds.load('coco/2017', data_dir=".")

"""
    Pre-Processing COCO 2017 from scratch.
    Download => process label and image file path => Pre-process and img load function => make tf dataset 
    ref: https://www.tensorflow.org/tutorials/text/image_captioning
"""

