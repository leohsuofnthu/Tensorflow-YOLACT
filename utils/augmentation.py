import numpy as np
import tensorflow as tf

from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
Ref: https://github.com/dbolya/yolact/blob/821e83047847b9b1faf21b03b0d7ad521508f8ee/utils/augmentations.py
"""


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks=None, boxes=None, labels=None):
        for transform in self.transforms:
            image, masks, boxes, labels = transform(image, masks, boxes, labels)
        return image, masks, boxes, labels


class ConvertFromInts(object):
    """Convert Iamge to tf.float32 and normalize to [0, 1]"""

    def __init__(self):
        ...

    def __call__(self, image, masks=None, boxes=None, labels=None):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, masks, boxes, labels


class Resize(object):
    def __int__(self):
        ...

    def __call__(self, image, masks, boxes, labels):
        ...


# normalize image fot input
class BackboneTransform(object):
    def __init__(self):
        ...

    def __call__(self, image, masks=None, boxes=None, labels=None):
        ...


class RandomBrightness(object):
    # Todo Make sure what is the range of image [0~255] or [0~1], and decide delta
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=self.delta)
        return image, masks, boxes, labels


class RandomContrast(object):
    # Todo Make sure what is the range of image [0~255] or [0~1], and decide lower, uppder
    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower
        assert lower >= 0
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_contrast(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels


class RandomSaturation(object):
    # Todo Make sure what is the range of image [0~255] or [0~1], and decide lower, uppder
    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower
        assert lower >= 0
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_saturation(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels


class RandomHue(object):
    # Todo Make sure what is the range of image [0~255] or [0~1], and decide lower, uppder
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_hue(image, max_delta=self.delta)
        return image, masks, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.actions = [
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, masks, boxes, labels):
        image, masks, boxes, labels = self.rand_brightness(image, masks, boxes, labels)
        # different order have different effect
        if tf.random.uniform([1]) > 0.5:
            photometric_distort = Compose(self.actions[:-1])
        else:
            photometric_distort = Compose(self.actions[1:])
        image, masks, boxes, labels = photometric_distort(image, masks, boxes, labels)
        return image, masks, boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        # exapnd the image with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            return image, masks, boxes, labels

        height, width, depth = image.get_shape()
        # expand 4 times at most
        ratio = tf.random.uniform([1], minval=1, maxval=4)
        # define the leftmost, topmost coordinate for putting original image to expanding image
        left = tf.random.uniform([1], minval=0, maxval=(width * ratio - width))
        top = tf.random.uniform([1], minval=0, maxval=(height * ratio - height))

        # padding the image, mask according to the left and top
        left_padding = int(left)
        top_padding = int(top)
        expand_width = int(width * ratio)
        expand_height = int(height * ratio)

        image = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(image, 0),
                                                        top_padding,
                                                        left_padding,
                                                        expand_height,
                                                        expand_width))
        masks = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(masks, 0),
                                                        top_padding,
                                                        left_padding,
                                                        expand_height,
                                                        expand_width))
        # fill mean value of image
        mean_mask = tf.cast((image == 0), image.dtype) + self.mean
        image = image + mean_mask

        # recalculate the bbox [ymin, xmin, ymax, xmax]
        boxes[0] = ((boxes[0] * height) + top) / expand_height
        boxes[1] = ((boxes[1] * width) + left) / expand_width
        boxes[2] = ((boxes[2] * height) + top) / expand_height
        boxes[3] = ((boxes[3] * width) + left) / expand_width

        return image, masks, boxes, labels


class RandomSampleCrop(object):
    def __init__(self):
        self.min_iou = [0, 0.1, 0.3, 0.7, 0.9, None]

    def __call__(self, image, masks, boxes=None, labels=None):
        # choose the min_object_covered value in self.sample_options
        idx = int(tf.random.uniform([1], minval=0, maxval=5.50))
        min_iou = self.min_iou[idx]
        if min_iou is None:
            return image, masks, boxes, labels

        # Geometric Distortions (img, bbox, mask)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=min_iou,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.1, 1.0),
            max_attempts=50)

        # the distort box is the area of the cropped image, original image will be [0, 0, 1, 1]
        distort_bbox = distort_bbox[0, 0]

        # cropped the image
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])

        # cropped the mask
        bbox_begin = tf.concat([[0], bbox_begin], axis=0)
        bbox_size = tf.concat([[-1], bbox_size], axis=0)
        cropped_masks = tf.slice(masks, bbox_begin, bbox_size)
        cropped_masks.set_shape([None, None, None, 1])

        # resize the scale of bboxes for cropped image
        v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
        boxes = boxes - v
        s = tf.stack([distort_bbox[2] - distort_bbox[0],
                      distort_bbox[3] - distort_bbox[1],
                      distort_bbox[2] - distort_bbox[0],
                      distort_bbox[3] - distort_bbox[1]])
        boxes = boxes / s

        # filter out
        scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], boxes.dtype), boxes)
        bool_mask = scores > 0.5
        classes = tf.boolean_mask(labels, bool_mask)
        bboxes = tf.boolean_mask(boxes, bool_mask)

        # deal with negative value of bbox
        bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

        cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)

        return cropped_image, cropped_masks, bboxes, classes


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
