import numpy as np
import tensorflow as tf

from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
Ref: https://github.com/dbolya/yolact/blob/821e83047847b9b1faf21b03b0d7ad521508f8ee/utils/augmentations.py
"""


class PhotometricDistort(object):
    def __init__(self):
        self.actions = []

    def __call__(self, image, masks, boxes, labels):
        ...


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        # exapnd the image with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            return image, masks, boxes, labels

        height, width, depth = tf.shape(image)
        ratio = ...
        left = ...
        top = ...


class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = {
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        }

    def __call__(self, image, masks, boxes=None, labels=None):
        height, width, _ = tf.shape(image)

        ...


class RandomMirror(object):
    def __int__(self):
        ...

    def __call__(self, image, masks, boxes, labels):
        # random mirroring with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)
            boxes = tf.stack([boxes[:, 0], 1 - boxes[:, 3],
                              boxes[:, 2], 1 - boxes[:, 1]], axis=-1)
        return image, masks, boxes, labels


def geometric_distortion(img, bboxes, masks, output_size, proto_output_size, classes):
    # Geometric Distortions (img, bbox, mask)
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=1,  # [0, 0.3, 0.5, 0.7, 0.9]
        aspect_ratio_range=(0.5, 2),
        area_range=(0.1, 1.0),
        max_attempts=100)
    # the distort box is the area of the cropped image, original image will be [0, 0, 1, 1]
    distort_bbox = distort_bbox[0, 0]
    # cropped the image
    cropped_image = tf.slice(img, bbox_begin, bbox_size)
    cropped_image.set_shape([None, None, 3])
    # cropped the mask
    bbox_begin = tf.concat([[0], bbox_begin], axis=0)
    bbox_size = tf.concat([[-1], bbox_size], axis=0)
    cropped_masks = tf.slice(masks, bbox_begin, bbox_size)
    cropped_masks.set_shape([None, None, None, 1])

    # resize the scale of bboxes for cropped image
    v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
    bboxes = bboxes - v
    s = tf.stack([distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1],
                  distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1]])
    bboxes = bboxes / s

    # filter out
    scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
    bool_mask = scores > 0.5
    classes = tf.boolean_mask(classes, bool_mask)
    bboxes = tf.boolean_mask(bboxes, bool_mask)

    # deal with negative value of bbox
    bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

    cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)
    # resize cropped to output size
    cropped_image = tf.image.resize(cropped_image, [output_size, output_size], method=tf.image.ResizeMethod.BILINEAR)

    return cropped_image, bboxes, cropped_masks, classes


"""Be Care for that, in here out input images have range [0, 1] instead of [0, 255]"""


def photometric_distortion(image):
    color_ordering = tf.random.uniform([1], minval=0, maxval=4)[0]

    if color_ordering < 1 and color_ordering > 0:
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_ordering < 2 and color_ordering > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    elif color_ordering < 3 and color_ordering > 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)

    return image


def horizontal_flip(image, bboxes, masks):
    # Random mirroring (img, bbox, mask)
    image = tf.image.flip_left_right(image)
    masks = tf.image.flip_left_right(masks)
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                       bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return image, bboxes, masks


def random_augmentation(img, bboxes, masks, output_size, proto_output_size, classes):
    # generate random
    FLAGS = tf.random.uniform([5], minval=0, maxval=1)
    tf.print("FLAGS:", FLAGS)
    # FLAG_GEO_DISTORTION = FLAGS[0]
    # FLAG_PHOTO_DISTORTION = FLAGS[1]
    # FLAG_HOR_FLIP = FLAGS[2]

    FLAG_GEO_DISTORTION = 0
    FLAG_PHOTO_DISTORTION = 0
    FLAG_HOR_FLIP = 0

    # Random Geometric Distortion (img, bboxes, masks)
    if FLAG_GEO_DISTORTION > 0.5:
        tf.print("GEO DISTORTION")
        img, bboxes, masks, classes = geometric_distortion(img, bboxes, masks, output_size, proto_output_size, classes)

    # Random Photometric Distortions (img)
    if FLAG_PHOTO_DISTORTION > 0.5:
        tf.print("PHOTO DISTORTION")
        img = photometric_distortion(img)

    if FLAG_HOR_FLIP > 0.5:
        tf.print("HOR FLIP")
        img, bboxes, masks = horizontal_flip(img, bboxes, masks)

    # resize masks to protosize
    masks = tf.image.resize(masks, [proto_output_size, proto_output_size],
                            method=tf.image.ResizeMethod.BILINEAR)
    masks = tf.cast(masks + 0.5, tf.int64)
    masks = tf.squeeze(masks)
    masks = tf.cast(masks, tf.float32)
    return img, bboxes, masks, classes
