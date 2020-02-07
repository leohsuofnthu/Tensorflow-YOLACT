import tensorflow as tf


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance.
     ref: https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/utils/input_utils.py
  """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def area_of_box(x):
    pass


def map_to_center_form(x):
    w = x[2] - x[0]
    h = x[3] - x[1]
    cx = x[0] + (w / 2)
    cy = x[1] + (h / 2)
    return tf.stack([cx, cy, w, h])


def map_to_point_form(x):
    xmin = x[0] - (x[2] / 2)
    ymin = x[1] - (x[3] / 2)
    xmax = x[0] + (x[2] / 2)
    ymax = x[1] + (x[3] / 2)
    return tf.stack([ymin, xmin, ymax, xmax])


# encode the gt and anchors to offset
def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = tf.math.log(x[2, 0] / x[2, 1])
    g_hat_h = tf.math.log(x[3, 0] / x[3, 1])
    return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])


# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(x):
    pass




def single_pair_iou(pred, target):
    # IOU of single pair of bbox, for the purpose in NMS, pairwise IOU is implemented within "Anchor" class
    """
    :param pred:
    :param target:
    :return:
    """

    pass
