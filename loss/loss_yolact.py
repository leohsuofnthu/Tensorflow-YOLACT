import tensorflow as tf


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

    def loss_yolact(self, pred, label, num_pos, anchors):
        """
        :param anchors:
        :param num_pos:
        :param label: labels dict from dataset
        :param pred:
        :return:
        """
        # extract the indices of positive example for each batch

        # _loss_location(pred_offset, gt_offset, positive_indices)

        # calculate the area of pred bounding boxes (anchors, pre_offset) for normalize the cls loss

        # _loss_class(pred_cls, gt_cls, positive_indices, neg_pos_ration)

        pass

    def _loss_location(self, pred_offset, gt_offset, positiveness):
        """
        :param pred_offset: [batch, num_anchor, 4]
        :param gt_offset (box_target): [batch, num_anchor, 4]
        :return:
        """
        # reshape the pred_offset from [batch, num_anchor, 4] => [batch*num_anchor, 4]
        pred_offset = pred_offset.reshape(-1, 4)
        # reshape the gt_offset from [batch, num_anchor, 4] => [batch*num_anchor, 4]
        gt_offset = gt_offset.reshape(-1, 4)
        # reshape the positiveness from [batch, num_anchor] => [batch*num_anchor, 1]
        positiveness = tf.expand_dims(positiveness, axis=-1).reshape(-1, 1)
        # extract the positive indices
        positive_indices = tf.where(positiveness > 0)

        # extract the positive example of pred of set according to the positive_indices
        # positive_pred = tf.gather_nd(..)
        # postive_gt = tf.gather_nd(..)
        # calculate the smoothL1(positive_pred, positive_gt) and return
        pass

    def _loss_class(self, pred_cls, gt_cls, num_cls, positive_indices, num_pos):
        """

        :param pred_cls:
        :param gt_cls:
        :param num_cls:
        :param positive_indices:
        :param neg_pos_ratio:
        :return:
        """
        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(-1, num_cls)

        # apply softmax on the pred_cls
        softmax_pred_cls = tf.nn.softmax(pred_cls, axis=-1)
        assert tf.reduce_sum(softmax_pred_cls[:, 0]) == 1

        # -log(softmax class 0)
        loss_minus_log_class0 = -1 * tf.math.log(softmax_pred_cls[:, 0])

        # eliminate the pos, neutral samples index

        # calculate the needed amount of  negative sample
        num_neg_needed = num_pos * self._neg_pos_ratio

        # take the first num_neg_needed idx in sort result and handle the situation if there are not enough neg

        # combine the indices of pos and neg sample, create the label for them

        # calculate Cross entropy loss and return

        pass

    def _loss_mask(self, proto_output, pred_mask_coef, gt_cls, gt_offset, gt_masks, positive_indices, max_masks_for_train):
        """
        loss of linear combination loss
        :return:
        """
        num_batch = tf.shape(proto_output)[0]
        tf.print("Batch_size:", num_batch)
        loss_mask = 0
        # Todo let s see if access by index is available
        for idx in tf.range(num_batch):
            # extract randomly postive sample in pred_mask_coef, gt_cls, gt_offset according to positive_indices
            # calculate sigmoid(pred_mask_coef_positive @ proto_output => [138, 138, num_pos])
            # create [138, 138, num_pos] correspond gt mask
            # iterate the each pair of pred_mask and gt_mask, calculate loss with cropped box
            # loss_mask += BCE(pred_mask, gt_mask)
            pass

        pass
