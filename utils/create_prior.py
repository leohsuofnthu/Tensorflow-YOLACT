from itertools import product
from math import sqrt
import numpy as np


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
        print("Create priors for f_size:%s", f_size)
        count_anchor = 0
        for j, i in product(range(f_size), range(f_size)):
            # +0.5 because priors are in center-size notation
            # print(j, i)
            f_k = img_size / (img_size / f_size)
            x = (i + 0.5) / f_k
            y = (j + 0.5) / f_k

            for ars in aspect_ratio:
                a = sqrt(ars)
                w = scale[idx] * a
                h = scale[idx] / a
                prior_boxes += [x, y, w, h]
                count_anchor += 1
                # print(x * img_size, y * img_size, w, h)
        num_anchors.append(count_anchor)
        print(f_size, count_anchor)
    output = np.asarray(prior_boxes).reshape(-1, 4)
    return num_anchors, output
