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

    def _pairwise_intersection(self, num_gt, gt_bbox):
        """
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # anchors = [num_anchors, 4]
        tf.print('num_anchors:', self.num_anchors)

        # intersection of every anchors and gts
        xy_min_anchors = tf.broadcast_to(tf.expand_dims(self.anchors[:, :2], axis=1), [self.num_anchors, num_gt])
        xy_min_gt = tf.broadcast_to(tf.expand_dims(num_gt[:, :2], axis=0), [self.num_anchors, num_gt])
        xy_max_anchors = tf.broadcast_to(tf.expand_dims(self.anchors[:, 2:], axis=1), [self.num_anchors, num_gt])
        xy_max_gt = tf.broadcast_to(tf.expand_dims(num_gt[:, 2:], axis=0), [self.num_anchors, num_gt])

        xy_min = tf.math.maximum(xy_min_anchors, xy_min_gt)
        xy_max = tf.math.minimum(xy_max_anchors, xy_max_gt)

        side_length = tf.clip_by_value((xy_max - xy_min), clip_value_min=0)
        intersection = side_length[:, 0] * side_length[:, 1]

        return intersection

    def _pairwise_iou(self, gt_bbox):
        """ˇ
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        num_gt = gt_bbox.shape[0]
        print('num_gt:', num_gt)

        # calculate areaA, areaB
        area_anchors = tf.broadcast_to(
            tf.expand_dims((self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1]),
                           axis=-1),
            [self.num_anchors, num_gt])

        area_gt = tf.broadcast_to(
            tf.expand_dims((gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1]), axis=0),
            [self.num_anchors, num_gt])

        # calculate A ∩ B (pairwise)
        inter = self._pairwise_intersection(num_gt, gt_bbox)

        # calculate A ∪ B
        union = area_anchors + area_gt - inter

        # IOU(jaccard overlap)
        iou = inter / union

        return iou

    def get_anchors(self):
        return self.anchors

    def matching(self, gt_bbox, gt_labels):
        """
        :param gt_bbox:
        :param gt_labels:
        :return:
        """
        # Todo how to handle negative coordinate of anchors
        # ignore the anchors that have negative value leave it blank
        # pairwise IoU
        # create class target
        # create loc target

        pass
