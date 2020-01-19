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
            num_anchors += count_anchor
            print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output

    def _pairwise_intersection(self, num_gt, gt_bbox):
        """
        ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py#L209
        :param gt_bbox: [num_obj, 4]
        :return:
        """

        # unstack the xmin, ymin, xmax, ymax
        xmin_anchor, ymin_anchor, xmax_anchor, ymax_anchor = tf.unstack(self.anchors, axis=-1)
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = tf.unstack(self.anchors, axis=-1)

        # calculate intersection
        all_pairs_max_xmin = tf.math.maximum(tf.expand_dims(xmin_anchor, axis=-1), tf.expand_dims(xmin_gt, axis=0))
        all_pairs_min_xmax = tf.math.minimum(tf.expand_dims(xmax_anchor, axis=-1), tf.expand_dims(xmax_gt, axis=0))
        all_pairs_max_ymin = tf.math.maximum(tf.expand_dims(ymin_anchor, axis=-1), tf.expand_dims(ymin_gt, axis=0))
        all_pairs_min_ymax = tf.math.minimum(tf.expand_dims(ymax_anchor, axis=-1), tf.expand_dims(ymax_gt, axis=0))
        intersect_heights = tf.math.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        intersect_widths = tf.math.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

        return intersect_heights * intersect_widths

    def _pairwise_iou(self, num_gt, gt_bbox):
        """ˇ
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        # calculate A ∩ B (pairwise)
        pairwise_inter = self._pairwise_intersection(num_gt=num_gt, gt_bbox=gt_bbox)

        # calculate areaA, areaB
        xmin_anchor, ymin_anchor, xmax_anchor, ymax_anchor = tf.unstack(self.anchors, axis=-1)
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = tf.unstack(self.anchors, axis=-1)

        area_anchor = (xmax_anchor-xmin_anchor) * (ymax_anchor - ymin_anchor)
        area_gt = (xmax_gt-xmin_anchor) * (xmax_gt - xmin_gt)

        # create same shape of matrix as intersection
        pairwise_area = tf.expand_dims(area_anchor, 0) + tf.expand_dims(area_gt, 1)

        # calculate A ∪ B
        pairwise_union = pairwise_area - pairwise_inter

        # IOU(Jaccard overlap) = intersection / union, there might be possible to have division by 0
        iou = pairwise_inter / pairwise_union

        print("finish pairwise IOU calculation...")

        return tf.where(
            tf.equal(pairwise_inter, 0.0),
            tf.zeros_like(pairwise_inter), pairwise_inter/pairwise_union)

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
        num_gt = gt_bbox.shape[0]
        print(tf.size(gt_bbox))
        print('num_gt:', num_gt)

        # pairwise IoU
        pairwise_iou = self._pairwise_iou(num_gt=num_gt, gt_bbox=gt_bbox)

        # assign the max overlap gt index for each anchor
        max_iou_for_anchors = tf.reduce_max(pairwise_iou, axis=-1)
        max_id_for_anchors = tf.math.argmax(pairwise_iou, axis=-1)

        # force the anchors which is the best matched of each gt to predict the correspond gt
        used_anchors = set()
        for idx in num_gt:
            max_id_for_gt = tf.math.argmax(pairwise_iou[:, idx], axis=0)
            # retrive the max anchor idx
            count = 0
            # check if is existed in set
            while max_id_for_gt[count] in used_anchors:
                count += 1
                print("anchor: %d has been assigned" % count)
            # yes, to next, no, add to set
            max_fit_anchor_id = max_id_for_gt[count]
            used_anchors.add(max_fit_anchor_id)

            max_id_for_anchors[max_fit_anchor_id].assign(idx)
            max_iou_for_anchors[max_fit_anchor_id].assign(pairwise_iou[max_fit_anchor_id, idx])

        # decide the anchors to be positive or negative based on the IoU and given threshold
        def _map_pos_match(x, pos, neg):
            if x < pos:
                return -1.
            elif x < neg:
                return 0.
            else:
                return 1.

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
        target_cls = tf.multiply(match_labels, match_positiveness)

        # create loc target
        map_loc = tf.map_fn(lambda x: gt_bbox[x], max_id_for_anchors, dtype=tf.float32)

        # convert to center form
        center_anchors = tf.map_fn(lambda x: map_to_center_form(x), self.anchors)
        center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)

        # calculate offset
        target_loc = tf.map_fn(lambda x: map_to_offset(x), tf.stack([center_gt, center_anchors], axis=-1))

        return target_cls, target_loc, max_id_for_anchors, match_positiveness
