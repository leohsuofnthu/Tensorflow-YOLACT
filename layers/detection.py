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
        top_k = tf.math.minimum(self.top_k, tf.size(candidate_ROI_idx))
        boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self.nms_threshold, top_k)
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200, second_threshold=False):
        scores, idx = tf.math.top_k(scores, k=top_k)
        tf.print("top k scores:", tf.shape(scores))
        tf.print("top k indices", tf.shape(idx))
        tf.print(idx[:20])

        num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]
        tf.print("num_classes:", num_classes)
        tf.print("num dets:", num_dets)

        tf.print("old boxes", tf.shape(boxes))

        boxes = tf.gather(boxes, idx)
        tf.print("new boxes", tf.shape(boxes))

        masks = tf.gather(masks, idx)
        tf.print("new masks", tf.shape(masks))

        iou = utils.jaccard(boxes, boxes)
        tf.print("iou", tf.shape(iou))

        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        tf.print("upper tri", upper_triangular[0])
        iou = upper_triangular - diag
        tf.print("iou", tf.shape(iou))
        tf.print("iou", iou[0])

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        tf.print("iou max", tf.shape(iou_max))
        tf.print("iou max", iou_max)

        idx_det = tf.where(iou_max <= iou_threshold)

        tf.print("idx det", tf.shape(idx_det))
        tf.print(idx_det)
        # Todo: second threshold, when to use
        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max))
        tf.print("classes", classes)
        classes = tf.gather_nd(classes, idx_det)
        tf.print("new_classes", tf.shape(classes))
        boxes = tf.gather_nd(boxes, idx_det)
        tf.print("new_boxes", tf.shape(boxes))
        masks = tf.gather_nd(masks, idx_det)
        tf.print("new_masks", tf.shape(masks))
        scores = tf.gather_nd(scores, idx_det)
        tf.print("new_scores", tf.shape(scores))
        tf.print(scores)

        # number of max detection = 100 (u can choose whatever u want)
        scores, idx = tf.math.top_k(scores, k=100)
        classes = tf.gather(classes, idx)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)
        scores = tf.gather(scores, idx)

        return boxes, masks, classes, scores
