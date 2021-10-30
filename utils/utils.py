"""
Ref:
https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/utils/input_utils.py
https://github.com/dbolya/yolact/blob/master/layers/box_utils.py
"""
import tensorflow as tf


# -----------------------------------------------------------------------------------------
# Functions used by loss/loss_yolact.py (mask loss)

# mapping from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
def map_to_center_form(x):
    h = x[:, 3] - x[:, 1]
    w = x[:, 2] - x[:, 0]
    cx = x[:, 0] + (w / 2.)
    cy = x[:, 1] + (h / 2.)
    return tf.stack([cx, cy, w, h], axis=-1)


def sanitize_coordinates(x1, x2, img_size, padding=0):
    x1 = tf.minimum(x1, x2)
    x2 = tf.maximum(x1, x2)
    x1 = tf.clip_by_value(x1 - padding, clip_value_min=0., clip_value_max=1000000.)
    x2 = tf.clip_by_value(x2 + padding, clip_value_min=0., clip_value_max=tf.cast(img_size, tf.float32))
    return x1, x2


# crop the prediction of mask so as to calculate the linear combination mask loss
def crop(pred, boxes):
    # pred [num_obj, 138, 138], gt [num_bboxes, 4]
    # sanitize coordination (make sure the bboxes are in range 0 <= x, y <= image size)
    shape_pred = tf.shape(pred)
    pred_w = shape_pred[0]
    pred_h = shape_pred[1]

    xmin, xmax = sanitize_coordinates(boxes[:, 0], boxes[:, 2], pred_w, padding=1)
    ymin, ymax = sanitize_coordinates(boxes[:, 1], boxes[:, 3], pred_h, padding=1)

    rows = tf.broadcast_to(tf.range(pred_w)[None, :, None], shape_pred)
    cols = tf.broadcast_to(tf.range(pred_h)[:, None, None], shape_pred)

    xmin = xmin[None, None, :]
    ymin = ymin[None, None, :]
    xmax = xmax[None, None, :]
    ymax = ymax[None, None, :]

    mask_left = (rows >= tf.cast(xmin, cols.dtype))
    mask_right = (rows <= tf.cast(xmax, cols.dtype))
    mask_bottom = (cols >= tf.cast(ymin, rows.dtype))
    mask_top = (cols <= tf.cast(ymax, rows.dtype))

    crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right),
                                    tf.math.logical_and(mask_bottom, mask_top))
    return pred * tf.cast(crop_mask, tf.float32)


# -----------------------------------------------------------------------------------------
# Functions used by layers/detection.py
def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def _convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(anchors, loc_pred):
    anchors = _convert_to_xywh(anchors)
    _box_variance = tf.convert_to_tensor(
        [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
    )
    loc_pred = loc_pred * _box_variance
    boxes = tf.concat(
        [
            loc_pred[:, :2] * anchors[:, 2:] + anchors[:, :2],
            tf.math.exp(loc_pred[:, 2:]) * anchors[:, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    return boxes_transformed


# -----------------------------------------------------------------------------------------
# Functions used by eval.py

def intersection(box_a, box_b):
    # unstack the xmin, ymin, xmax, ymax
    xmin_anchor, ymin_anchor, xmax_anchor, ymax_anchor = tf.unstack(box_a, axis=-1)
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = tf.unstack(box_b, axis=-1)

    # calculate intersection
    all_pairs_max_xmin = tf.math.maximum(tf.expand_dims(xmin_anchor, axis=-1), tf.expand_dims(xmin_gt, axis=1))
    all_pairs_min_xmax = tf.math.minimum(tf.expand_dims(xmax_anchor, axis=-1), tf.expand_dims(xmax_gt, axis=1))
    all_pairs_max_ymin = tf.math.maximum(tf.expand_dims(ymin_anchor, axis=-1), tf.expand_dims(ymin_gt, axis=1))
    all_pairs_min_ymax = tf.math.minimum(tf.expand_dims(ymax_anchor, axis=-1), tf.expand_dims(ymax_gt, axis=1))

    intersect_heights = tf.math.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    intersect_widths = tf.math.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths


def jaccard(box_a, box_b, is_crowd=False):
    # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
    # calculate A ∩ B (pairwise)
    pairwise_inter = intersection(box_a, box_b)
    # tf.print("pairwise inter", pairwise_inter)

    # calculate areaA, areaB
    xmin_a, ymin_a, xmax_a, ymax_a = tf.unstack(box_a, axis=-1)
    xmin_b, ymin_b, xmax_b, ymax_b = tf.unstack(box_b, axis=-1)

    area_a = (xmax_a - xmin_a) * (ymax_a - ymin_a)
    area_b = (xmax_b - xmin_b) * (ymax_b - ymin_b)

    # create same shape of matrix as intersection
    pairwise_area = tf.expand_dims(area_a, axis=-1) + tf.expand_dims(area_b, axis=1)

    # calculate A ∪ B
    pairwise_union = tf.expand_dims(area_a, axis=-1) if is_crowd else (pairwise_area - pairwise_inter)
    # tf.print("pairwise union", pairwise_union)

    # IOU(Jaccard overlap) = intersection / union, there might be possible to have division by 0
    return pairwise_inter / pairwise_union


def mask_iou(masks_a, masks_b, is_crowd=False):
    num_a = tf.shape(masks_a)[0]
    num_b = tf.shape(masks_b)[0]

    masks_a = tf.reshape(masks_a, (num_a, -1))
    masks_b = tf.reshape(masks_b, (num_b, -1))

    inter = tf.matmul(masks_a, masks_b, transpose_a=False, transpose_b=True)

    area_a = tf.expand_dims(tf.reduce_sum(masks_a, axis=-1), axis=-1)
    area_b = tf.expand_dims(tf.reduce_sum(masks_b, axis=-1), axis=0)

    union = area_a if is_crowd else (area_a + area_b - inter)

    return inter / union


def postprocess(detection, w, h, batch_idx, intepolation_mode="bilinear", crop_mask=True, score_threshold=0.5):
    """post process after detection layer"""
    dets = detection[batch_idx]
    dets = dets['detection']
    if dets is None:
        return None, None, None, None  # Warning, this is 4 copies of the same thing

    keep = tf.squeeze(tf.where(dets['score'] > score_threshold))
    for k in dets.keys():
        if k != 'proto':
            dets[k] = tf.gather(dets[k], keep)

    if tf.size(dets['score']) == 0:
        return None, None, None, None  # Warning, this is 4 copies of the same thing

    classes = dets['class']
    boxes = dets['box']
    scores = dets['score']
    masks = dets['mask']
    proto_pred = dets['proto']

    if tf.rank(masks) == 1:
        masks = tf.expand_dims(masks, axis=0)
        classes = tf.expand_dims(classes, axis=0)
        boxes = tf.expand_dims(boxes, axis=0)
        scores = tf.expand_dims(scores, axis=0)

    pred_mask = tf.linalg.matmul(proto_pred, masks, transpose_a=False, transpose_b=True)
    pred_mask = tf.nn.sigmoid(pred_mask)
    if crop_mask:
        masks = crop(pred_mask, boxes * float(tf.shape(pred_mask)[0] / w))
    masks = tf.transpose(masks, perm=[2, 0, 1])
    # intepolate to original size
    masks = tf.image.resize(tf.expand_dims(masks, axis=-1), [w, h],
                            method=intepolation_mode)
    # binarized the mask
    masks = tf.cast(masks + 0.5, tf.int64)
    masks = tf.squeeze(tf.cast(masks, tf.float32))
    # tf.print("masks after postprecessing", tf.shape(masks))

    return classes, scores, boxes, masks
