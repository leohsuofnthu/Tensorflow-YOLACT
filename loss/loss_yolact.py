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
        self._neg_pos_ration = neg_pos_ratio
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

    def _loss_location(self, pred_offset, gt_offset, positive_indices):
        """
        :param pred_offset: [num_positive_anchor, 4]
        :param gt_offset (box_target): [num_positive_anchor, 4]
        :return:
        """
        # extract the positive example of pred of set accroding to the positive_indices
        # calculate the smmoth L1 loss
        pass

    def _loss_class(self, pred_cls, gt_cls, positive_indices, neg_pos_ratio):
        """
        :param pred_cls:
        :param gt_cls:
        :return:
        """
        # extract the positive samples according to positive_indices calculate the needed negative sample
        # (num_pos * neg_pos_ration), and handle the situation if negative sample is not enough
        # extract the negative samples and concatenate with positive samples
        # calculate the loss
        # normalize the loss by corresponded area

        pass

    def _loss_mask(self, proto_output, pred_mask_coef, gt_cls, gt_masks, positive_indices, max_masks_for_train):
        """
        loss of linear combination loss
        :return:
        """
        # resize the gt_masks to proto output size (already done in parser)
        # extract the positive masks coef for train accroding to the positive_indices
        # check if the positive example <= max_masks_for_train and adjust
        # sigmoid(proto_output * positive mask coef) to get the masks
        # calculate the BCE loss of each sample accroding to the gt_cls
        pass
