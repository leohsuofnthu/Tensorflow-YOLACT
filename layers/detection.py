import tensorflow as tf

from utils import utils


class Detect(object):
    def __init__(self, anchors, num_cls, label_background, top_k, conf_threshold, nms_threshold, max_num_detection):
        self.num_cls = num_cls
        self.label_background = label_background
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.anchors = anchors
        self.max_num_detection = max_num_detection

    def __call__(self, prediction):
        loc_pred = prediction['pred_offset']
        cls_pred = prediction['pred_cls']
        mask_pred = prediction['pred_mask_coef']
        proto_pred = prediction['proto_out']
        num_batch = tf.shape(loc_pred)[0]

        # apply softmax to pred_cls
        cls_pred = tf.nn.softmax(cls_pred, axis=-1)
        cls_pred = tf.transpose(cls_pred, perm=[0, 2, 1])
        out = []
        for batch_idx in tf.range(num_batch):
            # add offset to anchors
            decoded_boxes = utils.map_to_bbox(self.anchors, loc_pred[batch_idx])
            # do detection, we ignore background label 0 here
            result = self._detection(cls_pred[batch_idx, 1:], decoded_boxes, mask_pred[batch_idx])
            if (result is not None) and (proto_pred is not None):
                result['proto'] = proto_pred[batch_idx]
            out.append({'detection': result})
        return out

    def _detection(self, cls_pred, decoded_boxes, mask_pred):
        cur_score = cls_pred
        # get scores and correspond class
        # cls_pred [20, 19248]
        conf_score = tf.reduce_max(cls_pred, axis=0)
        conf_score_id = tf.argmax(cls_pred, axis=0)

        # filter out the ROI that have conf score > confidence threshold
        candidate_ROI_idx = tf.squeeze(tf.where(conf_score > self.conf_threshold))

        # there might not have any score that over self.conf_threshold, no detection
        if tf.size(candidate_ROI_idx) == 0:
            return None
        else:
            scores = tf.gather(cur_score, candidate_ROI_idx, axis=-1)
            boxes = tf.gather(decoded_boxes, candidate_ROI_idx)
            masks_coef = tf.gather(mask_pred, candidate_ROI_idx)
            conf_score_id = tf.gather(conf_score_id, candidate_ROI_idx)

        # Fast NMS
        top_k = tf.math.minimum(self.top_k, tf.size(candidate_ROI_idx))
        boxes, masks, classes, scores = self._fast_nms(boxes, masks_coef, scores, self.nms_threshold, top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def _traditional_nms(self, boxes, masks, scores, classes, iou_threshold=0.5, top_k=200):
        if tf.rank(boxes) < 2:
            boxes = tf.expand_dims(boxes, axis=0)
            scores = tf.expand_dims(scores, axis=-1)
            masks = tf.expand_dims(masks, axis=0)
            classes = tf.expand_dims(classes, axis=0)
        selected_indices = tf.image.non_max_suppression(
            boxes, tf.reduce_max(scores, axis=0), top_k, iou_threshold)
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(tf.reduce_max(scores, axis=0), selected_indices)
        masks = tf.gather(masks, selected_indices)
        classes = tf.gather(classes, selected_indices)
        return boxes, masks, classes, scores

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200):
        if tf.rank(scores) == 1:
            scores = tf.expand_dims(scores, axis=-1)
            boxes = tf.expand_dims(boxes, axis=0)
            masks = tf.expand_dims(masks, axis=0)

        scores, idx = tf.math.top_k(scores, k=top_k)
        num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]
        boxes = tf.gather(boxes, idx, axis=0)
        masks = tf.gather(masks, idx, axis=0)
        iou = utils.jaccard(boxes, boxes)
        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        iou = upper_triangular - diag

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        idx_det = (iou_max <= iou_threshold)

        # second threshold
        # second_threshold = (iou_max <= self.conf_threshold)
        second_threshold = (scores > self.conf_threshold)
        idx_det = tf.where(tf.logical_and(idx_det, second_threshold) == True)
        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max))
        classes = tf.gather_nd(classes, idx_det)
        boxes = tf.gather_nd(boxes, idx_det)
        masks = tf.gather_nd(masks, idx_det)
        scores = tf.gather_nd(scores, idx_det)

        # number of max detection = 100 (u can choose whatever u want)
        max_num_detection = tf.math.minimum(self.max_num_detection, tf.size(scores))
        scores, idx = tf.math.top_k(scores, k=max_num_detection)

        # second threshold
        classes = tf.gather(classes, idx)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)

        return boxes, masks, classes, scores
