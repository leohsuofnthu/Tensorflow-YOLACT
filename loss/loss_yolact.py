import tensorflow as tf
import time
from utils import utils


class YOLACTLoss(object):

    def __init__(self, loss_weight_cls=1,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 loss_seg=1,
                 neg_pos_ratio=3,
                 max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_seg = loss_seg
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
        seg = pred['seg']

        # all label component
        cls_targets = label['cls_targets']
        box_targets = label['box_targets']
        positiveness = label['positiveness']
        bbox_norm = label['bbox_for_norm']
        masks = label['mask_target']
        max_id_for_anchors = label['max_id_for_anchors']
        classes = label['classes']
        num_obj = label['num_obj']

        # calculate num_pos
        loc_loss = self._loss_location(pred_offset, box_targets, positiveness) * self._loss_weight_box
        conf_loss = self._loss_class(pred_cls, cls_targets, num_classes, positiveness) * self._loss_weight_cls
        mask_loss = self._loss_mask(proto_out, pred_mask_coef, bbox_norm, masks, positiveness, max_id_for_anchors,
                                    max_masks_for_train=100) * self._loss_weight_mask
        seg_loss = self._loss_semantic_segmentation(seg, masks, classes, num_obj) * self._loss_weight_seg
        total_loss = loc_loss + conf_loss + mask_loss + seg_loss
        return loc_loss, conf_loss, mask_loss, seg_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, positiveness):

        positiveness = tf.expand_dims(positiveness, axis=-1)

        # get postive indices
        pos_indices = tf.where(positiveness == 1)
        pred_offset = tf.gather_nd(pred_offset, pos_indices[:, :-1])
        gt_offset = tf.gather_nd(gt_offset, pos_indices[:, :-1])

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]
        smoothl1loss = tf.keras.losses.Huber(delta=1., reduction=tf.losses.Reduction.NONE)
        loss_loc = tf.reduce_sum(smoothl1loss(gt_offset, pred_offset)) / tf.cast(num_pos, tf.float32)

        return loss_loc

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):

        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])

        # reshape gt_cls from [batch, num_anchor] => [batch * num_anchor, 1]
        gt_cls = tf.expand_dims(gt_cls, axis=-1)
        gt_cls = tf.reshape(gt_cls, [-1, 1])

        # reshape positiveness to [batch*num_anchor, 1]
        positiveness = tf.expand_dims(positiveness, axis=-1)
        positiveness = tf.reshape(positiveness, [-1, 1])
        pos_indices = tf.where(positiveness == 1)
        neg_indices = tf.where(positiveness == 0)

        # gather pos data, neg data separately
        pos_pred_cls = tf.gather(pred_cls, pos_indices[:, 0])
        pos_gt = tf.gather(gt_cls, pos_indices[:, 0])

        # calculate the needed amount of  negative sample
        num_pos = tf.shape(pos_gt)[0]
        num_neg_needed = num_pos * self._neg_pos_ratio

        neg_pred_cls = tf.gather(pred_cls, neg_indices[:, 0])
        neg_gt = tf.gather(gt_cls, neg_indices[:, 0])

        # apply softmax on the pred_cls
        neg_softmax = neg_pred_cls

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

        # loss
        loss_conf = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(target_labels, target_logits)) / tf.cast(
            num_pos, tf.float32)

        return loss_conf

    def _loss_mask(self, proto_output, pred_mask_coef, gt_bbox_norm, gt_masks, positiveness,
                   max_id_for_anchors, max_masks_for_train):

        shape_proto = tf.shape(proto_output)
        num_batch = shape_proto[0]
        loss_mask = 0.
        total_pos = 0
        for idx in tf.range(num_batch):
            # extract randomly postive sample in pred_mask_coef, gt_cls, gt_offset according to positive_indices
            proto = proto_output[idx]
            mask_coef = pred_mask_coef[idx]
            mask_gt = gt_masks[idx]
            bbox_norm = gt_bbox_norm[idx]
            pos = positiveness[idx]
            max_id = max_id_for_anchors[idx]

            pos_indices = tf.squeeze(tf.where(pos == 1))
            # tf.print("num_pos", tf.shape(pos_indices))
            """
            if tf.size(pos_indices) == 0:
                tf.print("detect no positive")
                continue
            """
            # Todo decrease the number pf positive to be 100
            # [num_pos, k]
            pos_mask_coef = tf.gather(mask_coef, pos_indices)
            pos_max_id = tf.gather(max_id, pos_indices)
            if tf.size(pos_indices) == 1:
                # tf.print("detect only one dim")
                pos_mask_coef = tf.expand_dims(pos_mask_coef, axis=0)
                pos_max_id = tf.expand_dims(pos_max_id, axis=0)
            total_pos += tf.size(pos_indices)
            # [138, 138, num_pos]
            pred_mask = tf.linalg.matmul(proto, pos_mask_coef, transpose_a=False, transpose_b=True)
            pred_mask = tf.transpose(pred_mask, perm=(2, 0, 1))

            # calculating loss for each mask coef correspond to each postitive anchor
            gt = tf.gather(mask_gt, pos_max_id)
            bbox = tf.gather(bbox_norm, pos_max_id)
            bbox_center = utils.map_to_center_form(bbox)
            area = bbox_center[:, -1] * bbox_center[:, -2]

            # crop the pred (not real crop, zero out the area outside the gt box)
            s = tf.nn.sigmoid_cross_entropy_with_logits(gt, pred_mask)
            s = utils.crop(s, bbox)
            loss = tf.reduce_sum(s, axis=[1, 2]) / area
            loss_mask += tf.reduce_sum(loss)

        loss_mask /= tf.cast(total_pos, tf.float32)
        return loss_mask

    def _loss_semantic_segmentation(self, pred_seg, mask_gt, classes, num_obj):

        shape_mask = tf.shape(mask_gt)
        num_batch = shape_mask[0]
        seg_shape = tf.shape(pred_seg)[1]
        loss_seg = 0.

        for idx in tf.range(num_batch):
            seg = pred_seg[idx]
            masks = mask_gt[idx]
            cls = classes[idx]
            objects = num_obj[idx]

            # seg shape (69, 69, num_cls)
            # resize masks from (100, 138, 138) to (100, 69, 69)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize(masks, [seg_shape, seg_shape], method=tf.image.ResizeMethod.BILINEAR)
            masks = tf.cast(masks + 0.5, tf.int64)
            masks = tf.squeeze(tf.cast(masks, tf.float32))

            # obj_mask shape (objects, 138, 138)
            obj_mask = masks[:objects]
            obj_cls = tf.expand_dims(cls[:objects], axis=-1)

            # create empty ground truth (138, 138, num_cls)
            seg_gt = tf.zeros_like(seg)
            seg_gt = tf.transpose(seg_gt, perm=(2, 0, 1))
            seg_gt = tf.tensor_scatter_nd_update(seg_gt, indices=obj_cls, updates=obj_mask)
            seg_gt = tf.transpose(seg_gt, perm=(1, 2, 0))
            loss_seg += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(seg_gt, seg))
        loss_seg = loss_seg / tf.cast(seg_shape, tf.float32) ** 2 / tf.cast(num_batch, tf.float32)

        return loss_seg
