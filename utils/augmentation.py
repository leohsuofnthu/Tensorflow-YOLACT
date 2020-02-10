import tensorflow as tf
from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
"""


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

    # resize the scale of bboxes for cropped image
    v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
    bboxes = bboxes - v
    s = tf.stack([distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1],
                  distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1]])
    bboxes = bboxes / s

    tf.print("original bbox", tf.shape(bboxes))
    scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
    bool_mask = scores > 0.5
    tf.print(bool_mask)
    classes = tf.boolean_mask(classes, bool_mask)
    tf.print(tf.shape(classes))
    bboxes = tf.boolean_mask(bboxes, bool_mask)
    tf.print(tf.shape(bboxes))
    masks = tf.boolean_mask(masks, bool_mask)
    tf.print(tf.shape(masks))

    # cropped the image
    cropped_image = tf.slice(img, bbox_begin, bbox_size)
    tf.print("crop img", tf.shape(cropped_image))
    cropped_image.set_shape([None, None, 3])

    # cropped the mask
    bbox_begin = tf.concat([[0], bbox_begin], axis=0)
    bbox_size = tf.concat([[-1], bbox_size], axis=0)
    cropped_mask = tf.slice(masks, bbox_begin, bbox_size)
    cropped_mask.set_shape([None, None, None, 1])
    tf.print("crop mask", tf.shape(cropped_mask))

    # resize cropped to output size
    img = tf.image.resize(img, [output_size, output_size], method=tf.image.ResizeMethod.BILINEAR)
    # resize mask, using nearest neighbor to make sure the mask still in binary
    masks = tf.image.resize(cropped_mask, [proto_output_size, proto_output_size], method=tf.image.ResizeMethod.BILINEAR)
    # binarize the mask
    masks = tf.cast(masks + 0.5, tf.int64)
    masks = tf.squeeze(masks)
    masks = tf.cast(masks, tf.float32)
    # Random mirroring (img, bbox, mask)

    # Photometric Distortions (img)

    # rescale to ResNet input (0~255) and use preprocess input function from tf keras ResNet 50

    return img, bboxes, masks, classes
