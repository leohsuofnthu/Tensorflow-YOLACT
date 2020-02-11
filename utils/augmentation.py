import numpy as np
import tensorflow as tf

from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
"""


def geometric_distortion(img, bboxes, masks, output_size, proto_output_size, classes):
    # Geometric Distortions (img, bbox, mask)
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=0.3,
        aspect_ratio_range=(0.5, 2),
        area_range=(0.1, 1.0),
        max_attempts=200,
        use_image_if_no_bounding_boxes=True)
    # the distort box is the area of the cropped image, original image will be [0, 0, 1, 1]
    distort_bbox = distort_bbox[0, 0]

    # cropped the image
    cropped_image = tf.slice(img, bbox_begin, bbox_size)
    tf.print("crop img", tf.shape(cropped_image))
    cropped_image.set_shape([None, None, 3])

    # cropped the mask
    bbox_begin = tf.concat([[0], bbox_begin], axis=0)
    bbox_size = tf.concat([[-1], bbox_size], axis=0)
    cropped_masks = tf.slice(masks, bbox_begin, bbox_size)
    cropped_masks.set_shape([None, None, None, 1])
    tf.print("crop mask", tf.shape(cropped_masks))

    # resize the scale of bboxes for cropped image
    v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
    bboxes = bboxes - v
    s = tf.stack([distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1],
                  distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1]])
    bboxes = bboxes / s

    # filter out
    tf.print("original bbox", tf.shape(bboxes))
    scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
    bool_mask = scores > 0.5
    tf.print(bool_mask)
    classes = tf.boolean_mask(classes, bool_mask)
    tf.print(tf.shape(classes))
    bboxes = tf.boolean_mask(bboxes, bool_mask)
    tf.print(tf.shape(bboxes))
    cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)
    tf.print("masks", tf.shape(cropped_masks))

    # resize cropped to output size
    cropped_image = tf.image.resize(cropped_image, [output_size, output_size], method=tf.image.ResizeMethod.BILINEAR)

    return cropped_image, bboxes, cropped_masks, classes


def photometric_distortion(image):
    color_ordering = np.random.randint(4, size=1)[0]
    if color_ordering == 0:
        tf.print("order 0")
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        tf.print("order 1")
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        tf.print("order 2")
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        tf.print("order 3")
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
        raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def horizontal_flip(image, bboxes, masks):
    # Random mirroring (img, bbox, mask)
    image = tf.image.flip_left_right(image)
    masks = tf.image.flip_left_right(masks)
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                       bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return image, bboxes, masks


def random_augmentation(img, bboxes, masks, output_size, proto_output_size, classes):
    """

    :param img:
    :param bbox:
    :param mask:
    :param output_size:
    :param proto_output_size:
    :return:
    """
    # normalize the bboxes
    bboxes = bboxes / output_size

    # generate random
    FLAGS = np.random.randint(2, size=3)
    FLAG_GEO_DISTORTION = 1
    FLAG_PHOTO_DISTORTION = 1
    FLAG_HOR_FLIP = 1

    # Random Geometric Distortion (img, bboxes, masks)
    if FLAG_GEO_DISTORTION:
        img, bboxes, masks, classes = geometric_distortion(img, bboxes, masks, output_size, proto_output_size, classes)
        tf.print("geo img", tf.shape(img))
        tf.print("geo bboxes", tf.shape(bboxes))
        tf.print("geo masks", tf.shape(masks))

    # Random Photometric Distortions (img)
    if FLAG_PHOTO_DISTORTION:
        img = photometric_distortion(img)
        tf.print("pho img", tf.shape(img))

    if FLAG_HOR_FLIP:
        if tf.size(bboxes) > 0:
            img, bboxes, masks = horizontal_flip(img, bboxes, masks)
            tf.print("flip img", tf.shape(img))
            tf.print("flip bboxes", tf.shape(bboxes))
            tf.print("flip masks", tf.shape(masks))

    # resize masks to protosize
    masks = tf.image.resize(masks, [proto_output_size, proto_output_size],
                            method=tf.image.ResizeMethod.BILINEAR)
    masks = tf.cast(masks + 0.5, tf.int64)
    masks = tf.squeeze(masks)
    masks = tf.cast(masks, tf.float32)

    # rescale to ResNet input (0~255) and use preprocess input function from tf keras ResNet 50
    # cropped_image = cropped_image * 255
    return img, bboxes, masks, classes
