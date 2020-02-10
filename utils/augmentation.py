import tensorflow as tf
from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
"""


def bbox_distortion():
    pass


def color_distortion(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
        raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def random_mirror():
    pass


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

    # Geometric Distortions (img, bbox, mask)
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=0.3,
        aspect_ratio_range=(0.9, 1.1),
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
    # resize mask, using nearest neighbor to make sure the mask still in binary
    cropped_masks = tf.image.resize(cropped_masks, [proto_output_size, proto_output_size],
                                    method=tf.image.ResizeMethod.BILINEAR)
    # binarize the mask
    cropped_masks = tf.cast(cropped_masks + 0.5, tf.int64)
    cropped_masks = tf.squeeze(cropped_masks)
    cropped_masks = tf.cast(cropped_masks, tf.float32)
    # Random mirroring (img, bbox, mask)
    cropped_image = tf.image.flip_left_right(cropped_image)
    cropped_masks = tf.image.flip_left_right(cropped_masks)
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                       bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)

    # Photometric Distortions (img)
    cropped_image = color_distortion(cropped_image)

    # rescale to ResNet input (0~255) and use preprocess input function from tf keras ResNet 50
    # cropped_image = cropped_image * 255

    return cropped_image, bboxes, cropped_masks, classes
