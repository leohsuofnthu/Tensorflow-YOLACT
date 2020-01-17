from itertools import product
from math import sqrt
import tensorflow as tf


class Anchor(object):

    def __init__(self, img_size, feature_map_size, aspect_ratio, scale):
        self.anchors = self._generate_anchors(img_size, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size, feature_map_size, aspect_ratio, scale):
        prior_boxes = []
        num_anchors = []
        for idx, f_size in enumerate(feature_map_size):
            print("Create priors for f_size:%s", f_size)
            count_anchor = 0
            for j, i in product(range(f_size), range(f_size)):
                f_k = img_size / (f_size + 1)
                x = f_k * (i + 1)
                y = f_k * (j + 1)
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a
                    h = scale[idx] / a
                    prior_boxes += [x, y, w, h]
                    count_anchor += 1
            num_anchors.append(count_anchor)
            print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output

    def _to_point_form(self, anchors):
        pass

    def to_area(self):
        pass

    def _pairwise_iou(self, gt_bbox):
        pass

    def get_anchors(self):
        pass

    def matching(self, gt_bbox, gt_labels):
        pass
