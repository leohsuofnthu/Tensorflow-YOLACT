import tensorflow as tf


def normalize_image(image):
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
    return tf.stack([xmin, ymin, xmax, ymax])


def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = tf.math.log(x[2, 0] / x[2, 1])
    g_hat_h = tf.math.log(x[3, 0] / x[3, 1])
    return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])

def single_pair_iou(pred, target):
    # IOU of single pair of bbox, for the purpose in NMS, pairwise IOU is implemented within "Anchor" class
    """
    :param pred:
    :param target:
    :return:
    """

    pass
