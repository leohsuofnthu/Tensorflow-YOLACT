import tensorflow as tf
from utils import utils


class Detect(object):
    def __init__(self, num_cls, label_background, top_k, conf_threshold, nms_threshold, anchors):
        self.num_cls = num_cls
        self.label_background = label_background
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.anchors = anchors

    def __call__(self, prediction):
        loc_pred = prediction['pred_offset']
        tf.print('loc pred', tf.shape(loc_pred))
        cls_pred = prediction['pred_cls']
        tf.print('cls pred', tf.shape(cls_pred))
        mask_pred = prediction['pred_mask_coef']
        tf.print('mask pred', tf.shape(mask_pred))
        proto_pred = prediction['proto_out']
        tf.print('proto pred', tf.shape(proto_pred))
        tf.print('anchors', tf.shape(self.anchors))
        out = []
        num_batch = tf.shape(loc_pred)[0]
        num_anchors = tf.shape(loc_pred)[1]
        tf.print("num batch:", num_batch)
        tf.print("num anchors:", num_anchors)

        # apply softmax to pred_cls
        cls_pred = tf.nn.softmax(cls_pred, axis=-1)
        tf.print("score", tf.shape(cls_pred))
        cls_pred = tf.transpose(cls_pred, perm=[0, 2, 1])
        tf.print("score", tf.shape(cls_pred))

        for batch_idx in tf.range(num_batch):
            # add offset to anchors
            decoded_boxes = utils.map_to_bbox(self.anchors, loc_pred[batch_idx])
            # tf.print(decoded_boxes)
            # do detection
            self._detection(batch_idx, cls_pred, decoded_boxes, mask_pred)
            pass

    def _detection(self, batch_idx, cls_pred, decoded_boxes, mask_pred):
        # we don't need to deal with background label
        cur_score = cls_pred[batch_idx, 1:, :]
        tf.print("cur score:", tf.shape(cur_score))
        conf_score = tf.math.reduce_max(cur_score, axis=0)
        tf.print("conf_score:", tf.shape(conf_score))

        # filter out the ROI that have conf score > confidence threshold
        candidate_ROI_idx = tf.squeeze(tf.where(conf_score > self.conf_threshold))
        tf.print("candidate_ROI", tf.shape(candidate_ROI_idx))

        if tf.size(candidate_ROI_idx) == 0:
            return None

        scores = tf.gather(cur_score, candidate_ROI_idx, axis=-1)
        tf.print("scores", tf.shape(scores))
        boxes = tf.gather(decoded_boxes, candidate_ROI_idx)
        tf.print("boxes", tf.shape(boxes))
        masks = tf.gather(mask_pred[batch_idx], candidate_ROI_idx)
        tf.print("masks", tf.shape(masks))

        # apply fast nms for final detection
        boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200, second_threshold=False):
        classes = 0
        return boxes, masks, classes, scores
