from itertools import product
from math import sqrt
import numpy as np


def make_priors(img_size, f_size, aspect_ratio, scale):
    """
    Create anchor boxes for each feature maps in [x, y, w, h], (x, y) is the center of anchor
    :param scales:
    :param size:
    :param aspect_ratio:
    :param scale:
    :return:
    """
    prior_boxes = []
    for j, i in product(range(f_size), range(f_size)):
        # +0.5 because priors are in center-size notation
        print(j, i)
        f_k = img_size/(img_size/f_size)
        x = (i + 0.5) / f_k
        y = (j + 0.5) / f_k


        for ars in aspect_ratio:
            a = sqrt(ars)
            w = scale * a
            h = scale / a
            prior_boxes += [x, y, w, h]
            print(x * img_size, y * img_size, w, h)
    return prior_boxes
