import tensorflow as tf
from utils import utils


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

        return self._loss_weight_box * loc_loss + self._loss_weight_cls * conf_loss + self._loss_weight_mask * mask_loss

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

        # calculate the smoothL1(positive_pred, positive_gt) and return
        smoothl1loss = tf.keras.losses.Huber(delta=0.5)
        loss_loc = tf.reduce_mean(smoothl1loss(gt_offset, pred_offset))
        tf.print("loss_loc:", loss_loc)
        return loss_loc

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):
        """

        :param pred_cls: [batch, num_anchor, num_cls]
        :param gt_cls: [batch, num_anchor, 1]
        :param num_cls:
        :return:
        """
        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])
        # tf.print("pred_cls:", tf.shape(pred_cls))

        # reshape gt_cls from [batch, num_anchor] => [batch * num_anchor, 1]
        gt_cls = tf.expand_dims(gt_cls, axis=-1)
        gt_cls = tf.reshape(gt_cls, [-1, 1])
        # tf.print("gt_cls:", tf.shape(gt_cls))

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

        # -log(softmax class 0)
        neg_minus_log_class0 = -1 * tf.math.log(neg_softmax[:, 0])

        # sort of -log(softmax class 0)
        neg_minus_log_class0_sort = tf.argsort(neg_minus_log_class0, direction="DESCENDING")
        # tf.print("neg_minus_log sort", tf.shape(neg_minus_log_class0_sort))

        # take the first num_neg_needed idx in sort result and handle the situation if there are not enough neg
        # Todo need to handle the situation if neg samples is not enough
        neg_indices_for_loss = neg_minus_log_class0_sort[:num_neg_needed]

        # combine the indices of pos and neg sample, create the label for them
        neg_pred_cls_for_loss = tf.gather(neg_pred_cls, neg_indices_for_loss)
        neg_gt_for_loss = tf.gather(neg_gt, neg_indices_for_loss)

        # calculate Cross entropy loss and return
        # concat positive and negtive data
        target_logits = tf.concat([pos_pred_cls, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.cast(tf.concat([pos_gt, neg_gt_for_loss], axis=0), tf.int64)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=num_cls)

        loss_conf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_labels, logits=target_logits))
        tf.print("loss_conf:", loss_conf)
        return loss_conf

    def _loss_mask(self, proto_output, pred_mask_coef, gt_bbox, gt_masks, positiveness,
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
        num_k = shape_proto[-1]
        # tf.print("Batch_size:", num_batch)
        # tf.print("K:", num_k)
        loss_mask = []
        for idx in tf.range(num_batch):
            # extract randomly postive sample in pred_mask_coef, gt_cls, gt_offset according to positive_indices
            proto = proto_output[idx]
            mask_coef = pred_mask_coef[idx]
            mask_gt = gt_masks[idx]
            pos = positiveness[idx]
            max_id = max_id_for_anchors[idx]

            # convert bbox to center form for area normalization in mask loss
            boxes_ct = tf.map_fn(lambda x: utils.map_to_center_form(x), gt_bbox[idx])

            pos_indices = tf.random.shuffle(tf.squeeze(tf.where(pos > 0)))
            num_pos = tf.size(pos_indices)
            # tf.print("num_pos =", num_pos)
            # Todo decrease the number pf positive to be 100
            # [num_pos, k]
            pos_mask_coef = tf.gather(mask_coef, pos_indices)
            pos_max_id = tf.gather(max_id, pos_indices)
            # [138, 138, num_pos]
            pred_mask = tf.linalg.matmul(proto, pos_mask_coef, transpose_a=False, transpose_b=True)
            # tf.print("shape of predmask:,", tf.shape(pred_mask))
            # tf.print(pos_max_id)

            # iterate the each pair of pred_mask and gt_mask, calculate loss with cropped box
            loss = 0
            bceloss = tf.keras.losses.BinaryCrossentropy()
            for num, value in enumerate(pos_max_id):
                gt = mask_gt[value]
                # read the w, h of original bbox and scale it to fit proto size
                box_w = boxes_ct[value][-2]
                box_h = boxes_ct[value][-1]
                pred = tf.nn.sigmoid(pred_mask[:, :, num])
                loss = loss + (bceloss(gt, pred) / box_w * box_h)

            loss_mask.append(loss)
        loss_mask = tf.math.reduce_mean(loss_mask)
        tf.print("loss_mask:", loss_mask)

        return loss_mask
