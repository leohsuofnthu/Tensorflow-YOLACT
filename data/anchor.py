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
        prior_boxes = []
        num_anchors = 0
        for idx, f_size in enumerate(feature_map_size):
            count_anchor = 0
            for j, i in product(range(f_size), range(f_size)):
                x = (i + 0.5) / f_size
                y = (j + 0.5) / f_size
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a / img_size
                    h = scale[idx] / a / img_size

                    # directly use point form here => [xmin, ymin, xmax, ymax]
                    xmin = x - (w / 2.)
                    ymin = y - (h / 2.)
                    xmax = x + (w / 2.)
                    ymax = y + (h / 2.)
                    prior_boxes += [xmin * img_size, ymin * img_size, xmax * img_size, ymax * img_size]
                count_anchor += 1
            num_anchors += count_anchor
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output

    def _convert_to_xywh(self, boxes):
        """Changes the box format to center, width and height.

        Arguments:
          boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
            representing bounding boxes where each box is of the format
            `[xmin, ymin, xmax, ymax]`.

        Returns:
          converted boxes with shape same as that of boxes.
        """
        return tf.concat(
            [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
            axis=-1,
        )

    def _pairwise_iou(self, gt_bbox):
        """
        ref: https://keras.io/examples/vision/retinanet/
        """
        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        lu = tf.maximum(self.anchors[:, None, :2], gt_bbox[:, :2])
        rd = tf.minimum(self.anchors[:, None, 2:], gt_bbox[:, 2:])
        intersection = tf.maximum(0.0, rd - lu)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

        anchorcenter = self._convert_to_xywh(self.anchors)
        gtcenter = self._convert_to_xywh(gt_bbox)
        anchors_area = anchorcenter[:, 2] * anchorcenter[:, 3]
        gts_area = gtcenter[:, 2] * gtcenter[:, 3]
        union_area = tf.maximum(
            anchors_area[:, None] + gts_area - intersection_area, 1e-8
        )
        return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    def _encode_boxes(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        _box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / _box_variance
        return box_target

    def get_anchors(self):
        return self.anchors

    def matching(self, gt_bbox, gt_labels, threshold_pos=0.5, threshold_neg=0.4):

        num_gt = tf.shape(gt_bbox)[0]
        pairwise_iou = self._pairwise_iou(gt_bbox=gt_bbox)

        # assign the max overlap gt index for each anchor
        max_iou_for_anchors = tf.reduce_max(pairwise_iou, axis=-1)
        max_id_for_anchors = tf.math.argmax(pairwise_iou, axis=-1)

        # force the anchors which is the best matched of each gt to predict the correspond gt
        forced_update_id = tf.cast(tf.range(0, num_gt), tf.int64)

        # force the iou over threshold for not wasting any training data
        forced_update_iou = tf.reduce_max(pairwise_iou, axis=0)
        # make sure the it won't be filtered even if under negative threshold
        forced_update_iou += (2 - forced_update_iou)
        forced_update_indice = tf.expand_dims(tf.math.argmax(pairwise_iou, axis=0), axis=-1)

        # assign the pair (the gt for priors to predict)
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, forced_update_indice, forced_update_iou)
        max_id_for_anchors = tf.tensor_scatter_nd_update(max_id_for_anchors, forced_update_indice, forced_update_id)

        # decide the anchors to be positive or negative based on the IoU and given threshold
        pos_iou = tf.where(max_iou_for_anchors > threshold_pos)
        neg_iou = tf.where(max_iou_for_anchors < threshold_neg)
        neu_iou = tf.where(
            tf.math.logical_and((max_iou_for_anchors <= threshold_pos), max_iou_for_anchors >= threshold_neg))

        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, pos_iou, tf.ones(tf.size(pos_iou)))
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, neg_iou, tf.zeros(tf.size(neg_iou)))
        max_iou_for_anchors = tf.tensor_scatter_nd_update(max_iou_for_anchors, neu_iou, -1 * tf.ones(tf.size(neu_iou)))
        match_positiveness = max_iou_for_anchors

        # create class target
        """
        element-wise multiplication of label[idx] and positiveness:
        1. positive sample will have correct label
        2. negative sample will have 0 * label[idx] = 0
        3. neural sample will have -1 * label[idx] = -1 * label[idx] 
        it can be useful to distinguish positive sample during loss calculation  
        """
        match_labels = tf.gather(gt_labels, max_id_for_anchors)
        target_cls = tf.multiply(tf.cast(match_labels, tf.float32), match_positiveness)

        # create loc target
        map_loc = tf.gather(gt_bbox, max_id_for_anchors)
        center_anchors = self._convert_to_xywh(self.anchors)
        center_gt = self._convert_to_xywh(map_loc)
        target_loc = self._encode_boxes(center_anchors, center_gt)

        return target_cls, target_loc, max_id_for_anchors, match_positiveness
