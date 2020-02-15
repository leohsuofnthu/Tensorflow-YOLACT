import tensorflow as tf
from utils import utils
import matplotlib.pyplot as plt


class YOLACTLoss(object):

    def __init__(self, loss_weight_cls=1,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 neg_pos_ratio=3,
                 max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train

    def __call__(self, pred, label, num_classes):
        """
        :param num_classes:
        :param anchors:
        :param label: labels dict from dataset
        :param pred:
        :return:
        """
        # all prediction component
        pred_cls = pred['pred_cls']
        pred_offset = pred['pred_offset']
        pred_mask_coef = pred['pred_mask_coef']
        proto_out = pred['proto_out']

        # all label component
        cls_targets = label['cls_targets']
        box_targets = label['box_targets']
        positiveness = label['positiveness']
        bbox_norm = label['bbox_for_norm']
        masks = label['mask_target']
        max_id_for_anchors = label['max_id_for_anchors']

        loc_loss = self._loss_location(pred_offset, box_targets, positiveness)
        conf_loss = self._loss_class(pred_cls, cls_targets, num_classes, positiveness)
        mask_loss = self._loss_mask(proto_out, pred_mask_coef, bbox_norm, masks, positiveness, max_id_for_anchors,
                                    max_masks_for_train=100)

        total_loss = self._loss_weight_box * loc_loss + self._loss_weight_cls * conf_loss + self._loss_weight_mask * mask_loss

        return loc_loss, conf_loss, mask_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, positiveness):
        """
        :param pred_offset: [batch, num_anchor, 4]
        :param gt_offset (box_target): [batch, num_anchor, 4]
        :return:
        """
        positiveness = tf.expand_dims(positiveness, axis=-1)

        # get postive indices
        pos_indices = tf.where(positiveness > 0)

        pred_offset = tf.gather_nd(pred_offset, pos_indices[:, :-1])
        gt_offset = tf.gather_nd(gt_offset, pos_indices[:, :-1])

        # check if there is nan in pred_offset, gt_offset
        tf.debugging.check_numerics(pred_offset, message="pred_offset contains invalid value")
        tf.debugging.check_numerics(gt_offset, message="gt_offset contains invalid value")

        # calculate the smoothL1(positive_pred, positive_gt) and return
        smoothl1loss = tf.keras.losses.Huber(delta=0.5, reduction=tf.losses.Reduction.NONE)
        loss_loc = tf.reduce_sum(smoothl1loss(gt_offset, pred_offset)) / tf.cast(tf.size(pos_indices), tf.float32)
        tf.print("loc loss:", loss_loc)
        return loss_loc

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):
        """

        :param pred_cls: [batch, num_anchor, num_cls]
        :param gt_cls: [batch, num_anchor, 1]
        :param num_cls:
        :return:
        """
        # check if there is nan in pred_cls, gt_cls
        tf.debugging.check_numerics(pred_cls, message="pred_cls contains invalid value")
        tf.debugging.check_numerics(gt_cls, message="gt_cls contains invalid value")

        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])

        # reshape gt_cls from [batch, num_anchor] => [batch * num_anchor, 1]
        gt_cls = tf.expand_dims(gt_cls, axis=-1)
        gt_cls = tf.reshape(gt_cls, [-1, 1])

        # reshape positiveness to [batch*num_anchor, 1]
        positiveness = tf.expand_dims(positiveness, axis=-1)
        positiveness = tf.reshape(positiveness, [-1, 1])
        pos_indices = tf.where(positiveness > 0)
        neg_indices = tf.where(positiveness == 0)

        # calculate the needed amount of  negative sample
        num_pos = tf.size(pos_indices[:, 0])
        # tf.print("num_pos = ", num_pos)
        num_neg_needed = num_pos * self._neg_pos_ratio
        # tf.print("num_neg = ", num_neg_needed)

        # gather pos data, neg data separately
        pos_pred_cls = tf.gather(pred_cls, pos_indices[:, 0])
        pos_gt = tf.gather(gt_cls, pos_indices[:, 0])

        neg_pred_cls = tf.gather(pred_cls, neg_indices[:, 0])
        neg_gt = tf.gather(gt_cls, neg_indices[:, 0])

        # apply softmax on the pred_cls
        neg_softmax = tf.nn.softmax(neg_pred_cls)
        tf.debugging.check_numerics(neg_softmax, message="neg_softmax contains invalid value")
        # -log(softmax class 0)
        neg_minus_log_class0 = -1 * tf.math.log(neg_softmax[:, 0])

        # sort of -log(softmax class 0)
        neg_minus_log_class0_sort = tf.argsort(neg_minus_log_class0, direction="DESCENDING")

        # take the first num_neg_needed idx in sort result and handle the situation if there are not enough neg
        neg_indices_for_loss = neg_minus_log_class0_sort[:num_neg_needed]

        # combine the indices of pos and neg sample, create the label for them
        neg_pred_cls_for_loss = tf.gather(neg_pred_cls, neg_indices_for_loss)
        neg_gt_for_loss = tf.gather(neg_gt, neg_indices_for_loss)

        # calculate Cross entropy loss and return
        # concat positive and negtive data
        target_logits = tf.concat([pos_pred_cls, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.cast(tf.concat([pos_gt, neg_gt_for_loss], axis=0), tf.int64)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=num_cls)

        loss_conf = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_labels, logits=target_logits)) / (
                        tf.cast(tf.size(pos_indices), tf.float32))
        tf.debugging.check_numerics(loss_conf, message="loss_conf contains invalid value")
        tf.print("conf loss:", loss_conf)
        return loss_conf

    def _loss_mask(self, proto_output, pred_mask_coef, gt_bbox_norm, gt_masks, positiveness,
                   max_id_for_anchors, max_masks_for_train):
        """

        :param proto_output:
        :param pred_mask_coef:
        :param gt_bbox:
        :param gt_masks:
        :param positiveness:
        :param max_id_for_anchors:
        :param max_masks_for_train:
        :return:
        """
        shape_proto = tf.shape(proto_output)
        num_batch = shape_proto[0]
        loss_mask = []
        for idx in tf.range(num_batch):
            # extract randomly postive sample in pred_mask_coef, gt_cls, gt_offset according to positive_indices
            proto = proto_output[idx]
            mask_coef = pred_mask_coef[idx]
            mask_gt = gt_masks[idx]
            bbox_norm = gt_bbox_norm[idx]
            pos = positiveness[idx]
            max_id = max_id_for_anchors[idx]

            pos_indices = tf.random.shuffle(tf.squeeze(tf.where(pos > 0)))
            # tf.print("num_pos =", num_pos)
            # Todo decrease the number pf positive to be 100
            # [num_pos, k]
            pos_mask_coef = tf.gather(mask_coef, pos_indices)
            pos_max_id = tf.gather(max_id, pos_indices)

            if tf.size(pos_indices) == 0:
                tf.print("detect no positive")
                continue
            elif tf.size(pos_indices) == 1:
                tf.print("detect only one dim")
                pos_mask_coef = tf.expand_dims(pos_mask_coef, axis=0)
                pos_max_id = tf.expand_dims(pos_max_id, axis=0)

            # [138, 138, num_pos]
            pred_mask = tf.linalg.matmul(proto, pos_mask_coef, transpose_a=False, transpose_b=True)

            # iterate the each pair of pred_mask and gt_mask, calculate loss with cropped box
            loss = 0
            bceloss = tf.keras.losses.BinaryCrossentropy()
            for num, value in enumerate(pos_max_id):
                gt = mask_gt[value]
                bbox = bbox_norm[value]
                bbox_center = utils.map_to_center_form(bbox)
                area = bbox_center[-1] * bbox_center[-2]
                ymin, xmin, ymax, xmax = tf.unstack(bbox)
                ymin = tf.cast(tf.math.floor(ymin), tf.int64)
                xmin = tf.cast(tf.math.floor(xmin), tf.int64)
                ymax = tf.cast(tf.math.ceil(ymax), tf.int64)
                xmax = tf.cast(tf.math.ceil(xmax), tf.int64)
                # read the w, h of original bbox and scale it to fit proto size
                pred = pred_mask[:, :, num]
                loss = loss + ((bceloss(gt[ymin:ymax, xmin:xmax], pred[ymin:ymax, xmin:xmax])) / area)
                # plt.figure()
                # plt.imshow(gt[ymin:ymax, xmin:xmax])
            # plt.show()
            loss_mask.append(loss / tf.cast(tf.size(num_batch), tf.float32))
        loss_mask = tf.math.reduce_sum(loss_mask)
        tf.print("mask loss:", loss_mask)
        return loss_mask

    def _loss_semantic_segmentation(self):
        # implemented after training sucessfully
        pass
