import tensorflow as tf
from utils import utils


class Detect(object):
    def __init__(self, anchors, num_cls, label_background, top_k, conf_threshold, nms_threshold):
        self.num_cls = num_cls
        self.label_background = label_background
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.anchors = anchors

    def __call__(self, prediction):
        loc_pred = prediction['pred_offset']
        # tf.print('loc pred', tf.shape(loc_pred))
        cls_pred = prediction['pred_cls']
        # tf.print('cls pred', tf.shape(cls_pred))
        mask_pred = prediction['pred_mask_coef']
        # tf.print('mask pred', tf.shape(mask_pred))
        proto_pred = prediction['proto_out']
        # tf.print('proto pred', tf.shape(proto_pred))
        # tf.print('anchors', tf.shape(self.anchors))
        num_batch = tf.shape(loc_pred)[0]
        # tf.print("num batch:", num_batch)
        # tf.print("num anchors:", num_anchors)

        # apply softmax to pred_cls
        cls_pred = tf.nn.softmax(cls_pred, axis=-1)
        # tf.print("score", tf.shape(cls_pred))
        cls_pred = tf.transpose(cls_pred, perm=[0, 2, 1])
        # tf.print("score", tf.shape(cls_pred))

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

        # get scores and correspond class
        conf_score = tf.math.reduce_max(cls_pred, axis=0)
        conf_score_id = tf.argmax(cls_pred, axis=0)
        # tf.print("conf_score:", tf.shape(conf_score))

        # filter out the ROI that have conf score > confidence threshold
        candidate_ROI_idx = tf.squeeze(tf.where(conf_score > self.conf_threshold))
        # tf.print("candidate_ROI", candidate_ROI_idx)

        # there might not have any score that over self.conf_threshold, no detection
        if tf.size(candidate_ROI_idx) == 0:
            return None
        else:
            # tf.print('original score', tf.shape(cur_score))
            scores = tf.gather(conf_score, candidate_ROI_idx)
            # scores = tf.gather(conf_score, candidate_ROI_idx)
            classes = tf.gather(conf_score_id, candidate_ROI_idx)
            # tf.print("scores", tf.shape(scores))
            boxes = tf.gather(decoded_boxes, candidate_ROI_idx)
            # tf.print("boxes", tf.shape(boxes))
            masks = tf.gather(mask_pred, candidate_ROI_idx)
            # tf.print("masks", tf.shape(masks))

        # Fast NMS
        # tf.print("before fastnms score", scores)
        top_k = tf.math.minimum(self.top_k, tf.size(candidate_ROI_idx))
        boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self.nms_threshold, top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200, second_threshold=False):

        scores, idx = tf.math.top_k(scores, k=top_k)
        num_classes = tf.shape(idx)[0]
        boxes = tf.gather(boxes, idx, axis=0)
        masks = tf.gather(masks, idx, axis=0)
        iou = utils.jaccard(boxes, boxes)

        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        iou = upper_triangular - diag

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        idx_det = tf.where(iou_max <= iou_threshold)

        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max)) + 1
        classes = tf.gather_nd(classes, idx_det)
        boxes = tf.gather_nd(boxes, idx_det)
        masks = tf.gather_nd(masks, idx_det)
        scores = tf.gather_nd(scores, idx_det)

        # number of max detection = 100 (u can choose whatever u want)
        max_num_detection = tf.math.minimum(self.top_k, tf.size(scores))
        scores, idx = tf.math.top_k(scores, k=max_num_detection)
        classes = tf.gather(classes, idx)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)

        # second threshold
        positive_det = tf.squeeze(tf.where(scores > iou_threshold))
        scores = tf.gather(scores, positive_det)
        classes = classes[:tf.size(scores)]
        boxes = boxes[:tf.size(scores)]
        masks = masks[:tf.size(scores)]

        return boxes, masks, classes, scores

    def _cc_fast_nms(self):
        """cross class FastNMS"""
        ...

    def _nms(self):
        """original NMS"""
        """
               #　Normal NMS
               selected_indices = tf.image.non_max_suppression(boxes, scores, 100, 0.1)
               boxes = tf.gather(boxes, selected_indices)
               scores = tf.gather(scores, selected_indices)
               masks = tf.gather(masks, selected_indices)
               classes = tf.gather(classes, selected_indices)

               tf.print("predicted boxes shape", tf.shape(boxes))
               tf.print("predicted scores shape", tf.shape(scores))
               tf.print("predicted masks shape", tf.shape(masks))
               tf.print("predicted classes shape", tf.shape(classes))
               """
        ...
