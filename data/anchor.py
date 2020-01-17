from itertools import product
from math import sqrt
import tensorflow as tf


# Can generate one instance only when creating the model
class Anchor(object):

    def __init__(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.num_anchors, self.anchors = self._generate_anchors(img_size, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
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
                f_k = img_size / (f_size + 1)
                x = f_k * (i + 1)
                y = f_k * (j + 1)
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a
                    h = scale[idx] / a
                    # directly use point form here => [xmin, ymin, xmax, ymax]
                    prior_boxes += [x - (w / 2), y - (h / 2), x + (w / 2), y + (h / 2)]
                    count_anchor += 1
            num_anchors.append(count_anchor)
            print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output

    def _pairwise_intersection(self, gt_bbox):
        """
        :param gt_bbox:
        :return:
        """
        # num_gt
        # num_anchor

        # area of each anchor
        # area of each gt

        # intersection of every anchors and gts

        pass

    def _pairwise_iou(self, gt_bbox):
        """ˇ
        :param gt_bbox:
        :return:
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)

        # calculate A ∩ B (pairwise)

        pass

    def get_anchors(self):
        return self.anchors

    def matching(self, gt_bbox, gt_labels):
        """
        :param gt_bbox:
        :param gt_labels:
        :return:
        """
        # ignore the anchors that have negative value leave it blank
        # pairwise IoU
        # create class target
        # create loc target

        pass
