import tensorflow as tf


class YOLACTLoss(object):

    def __init__(self, loss_weight_cls=1, loss_weight_box=1.5, loss_weight_mask=6.125):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask

    def _loss_location(self, pred_offset, gt_offset):
        """
        :param pred_offset: [num_positive_anchor, 4]
        :param gt_offset: [num_positive_anchor, 4]
        :return:
        """
        pass

    def _loss_class(self, pred_cls, gt_cls):
        """
        :param pred_cls:
        :param gt_cls:
        :return:
        """
        pass

    def _loss_mask(self):
        """
        loss of linear combination loss
        :return:
        """
        pass
