from itertools import product
from math import sqrt

import tensorflow as tf


def make_priors(img_size, feature_map_size, aspect_ratio, scale):
    """
    Create anchor boxes for each feature maps in [x, y, w, h], (x, y) is the center of anchor
    :param feature_map_size:
    :param img_size:
    :param aspect_ratio:
    :param scale:
    :return:
    """
    prior_boxes = []
    num_anchors = []
    for idx, f_size in enumerate(feature_map_size):
        # print("Create priors for f_size:%s", f_size)
        count_anchor = 0
        for j, i in product(range(f_size), range(f_size)):
            f_k = img_size / (f_size + 1)
            x = f_k * (i + 1)
            y = f_k * (j + 1)
            for ars in aspect_ratio:
                a = sqrt(ars)
                w = scale[idx] * a
                h = scale[idx] / a
                prior_boxes += [x - (w / 2), y - (h / 2), x + (w / 2), y + (h / 2)]
                count_anchor += 1
        num_anchors.append(count_anchor)
        # print(f_size, count_anchor)
    output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
    return num_anchors, output
