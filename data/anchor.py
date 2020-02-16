from itertools import product
from math import sqrt
from utils.utils import map_to_center_form, map_to_offset
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
        num_anchors = 0
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
                    # directly use point form here => [ymin, xmin, ymax, xmax]
                    ymin = max(0, y - (h / 2))
                    xmin = max(0, x - (w / 2))
                    ymax = min(img_size, y + (h / 2))
                    xmax = min(img_size, x + (w / 2))
                    prior_boxes += [ymin, xmin, ymax, xmax]
                    count_anchor += 1
            num_anchors += count_anchor
            # print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output

    def _pairwise_intersection(self, gt_bbox):
        """
        ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """

        # unstack the ymin, xmin, ymax, xmax
        ymin_anchor, xmin_anchor, ymax_anchor, xmax_anchor = tf.unstack(self.anchors, axis=-1)
        ymin_gt, xmin_gt, ymax_gt, xmax_gt = tf.unstack(gt_bbox, axis=-1)

        # calculate intersection
        all_pairs_max_xmin = tf.math.maximum(tf.expand_dims(xmin_anchor, axis=-1), tf.expand_dims(xmin_gt, axis=0))
        all_pairs_min_xmax = tf.math.minimum(tf.expand_dims(xmax_anchor, axis=-1), tf.expand_dims(xmax_gt, axis=0))
        all_pairs_max_ymin = tf.math.maximum(tf.expand_dims(ymin_anchor, axis=-1), tf.expand_dims(ymin_gt, axis=0))
        all_pairs_min_ymax = tf.math.minimum(tf.expand_dims(ymax_anchor, axis=-1), tf.expand_dims(ymax_gt, axis=0))
        intersect_heights = tf.math.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        intersect_widths = tf.math.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

        return intersect_heights * intersect_widths

    def _pairwise_iou(self, gt_bbox):
        """ˇ
         ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        # calculate A ∩ B (pairwise)
        pairwise_inter = self._pairwise_intersection(gt_bbox=gt_bbox)

        # calculate areaA, areaB
        ymin_anchor, xmin_anchor, ymax_anchor, xmax_anchor = tf.unstack(self.anchors, axis=-1)
        ymin_gt, xmin_gt, ymax_gt, xmax_gt = tf.unstack(gt_bbox, axis=-1)

        area_anchor = (xmax_anchor - xmin_anchor) * (ymax_anchor - ymin_anchor)
        area_gt = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)

        # create same shape of matrix as intersection
        pairwise_area = tf.expand_dims(area_anchor, axis=-1) + tf.expand_dims(area_gt, axis=0)

        # calculate A ∪ B
        pairwise_union = pairwise_area - pairwise_inter

        # IOU(Jaccard overlap) = intersection / union, there might be possible to have division by 0
        return tf.where(
            tf.equal(pairwise_union, 0.0),
            tf.zeros_like(pairwise_union), pairwise_inter / pairwise_union)

    def get_anchors(self):
        return self.anchors

    def matching(self, threshold_pos, threshold_neg, gt_bbox, gt_labels):
        """
        :param threshold_neg:
        :param threshold_pos:
        :param gt_bbox:
        :param gt_labels:
        :return:
        """
        num_gt = tf.shape(gt_bbox)[0]
        # tf.print("num gt", num_gt)
        # pairwise IoU
        pairwise_iou = self._pairwise_iou(gt_bbox=gt_bbox)
        # tf.print("iou", pairwise_iou)

        # assign the max overlap gt index for each anchor
        max_iou_for_anchors = tf.reduce_max(pairwise_iou, axis=-1)
        max_id_for_anchors = tf.math.argmax(pairwise_iou, axis=-1)

        # force the anchors which is the best matched of each gt to predict the correspond gt
        forced_update_id = tf.cast(tf.range(0, num_gt), tf.int64)
        forced_update_iou = tf.reduce_max(pairwise_iou, axis=0)
        forced_update_indice = tf.expand_dims(tf.math.argmax(pairwise_iou, axis=0), axis=-1)
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, forced_update_indice, forced_update_iou)
        max_id_for_anchors = tf.tensor_scatter_nd_update(max_id_for_anchors, forced_update_indice, forced_update_id)

        # decide the anchors to be positive or negative based on the IoU and given threshold
        def _map_pos_match(x, pos, neg):
            p = 1.
            if x < pos:
                p = -1.
            if x < neg:
                p = 0.
            return p

        match_positiveness = tf.map_fn(lambda x: _map_pos_match(x, threshold_pos, threshold_neg)
                                       , max_iou_for_anchors)

        # create class target
        # map idx to label[idx]
        match_labels = tf.map_fn(lambda x: gt_labels[x], max_id_for_anchors)

        """
        element-wise multiplication of label[idx] and positiveness:
        1. positive sample will have correct label
        2. negative sample will have 0 * label[idx] = 0
        3. neural sample will have -1 * label[idx] = -1 * label[idx] 
        it can be useful to distinguish positive sample during loss calculation  
        """

        target_cls = tf.multiply(tf.cast(match_labels, tf.float32), match_positiveness)

        # create loc target
        map_loc = tf.map_fn(lambda x: gt_bbox[x], max_id_for_anchors, dtype=tf.float32)

        # convert to center form

        # center_anchors = tf.map_fn(lambda x: map_to_center_form(x), self.anchors)
        w = self.anchors[:, 2] - self.anchors[:, 0]
        h = self.anchors[:, 3] - self.anchors[:, 1]
        center_anchors = tf.stack([self.anchors[:, 0] + (w / 2), self.anchors[:, 1] + (h / 2), w, h])

        # center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)
        w = map_loc[:, 2] - map_loc[:, 0]
        h = map_loc[:, 3] - map_loc[:, 1]
        center_gt = tf.stack([map_loc[:, 0] + (w / 2), map_loc[:, 1] + (h / 2), w, h])

        # calculate offset
        # target_loc = tf.map_fn(lambda x: map_to_offset(x), tf.stack([center_gt, center_anchors], axis=-1))
        g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]) / center_anchors[:, 2]
        g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]) / center_anchors[:, 3]
        g_hat_w = tf.math.log(center_anchors[:, 2] / center_gt[:, 2])
        g_hat_h = tf.math.log(center_anchors[:, 3] / center_gt[:, 3])
        target_loc = tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])

        return target_cls, target_loc, max_id_for_anchors, match_positiveness
