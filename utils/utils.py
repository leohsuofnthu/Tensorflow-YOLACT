import tensorflow as tf


def bboxes_intersection(bbox_ref, bboxes):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """

    # Should be more efficient to first transpose.
    bboxes = tf.transpose(bboxes)
    bbox_ref = tf.transpose(bbox_ref)
    # Intersection bbox and volume.
    int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
    int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
    int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
    int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    # Volumes.
    inter_vol = h * w
    bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])

    return tf.where(
        tf.equal(bboxes_vol, 0.0),
        tf.zeros_like(inter_vol), inter_vol / bboxes_vol)


def normalize_image(image, offset=(0.485, 0.456, 0.406), scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance.
     ref: https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/utils/input_utils.py
  """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    image *= 255
    return image


# mapping from [ymin, xmin, ymax, xmax] to [cx, cy, w, h]
def map_to_center_form(x):
    h = x[:, 2] - x[:, 0]
    w = x[:, 3] - x[:, 1]
    cy = x[:, 0] + (h / 2)
    cx = x[:, 1] + (w / 2)
    return tf.stack([cx, cy, w, h], axis=-1)


# encode the gt and anchors to offset
def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = tf.math.log(x[2, 0] / x[2, 1])
    g_hat_h = tf.math.log(x[3, 0] / x[3, 1])
    return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])


# crop the prediction of mask so as to calculate the linear combination mask loss
def crop(pred, boxes):
    pred_shape = tf.shape(pred)
    w = tf.cast(tf.range(pred_shape[1]), tf.float32)
    h = tf.expand_dims(tf.cast(tf.range(pred_shape[2]), tf.float32), axis=-1)

    cols = tf.broadcast_to(w, pred_shape)
    rows = tf.broadcast_to(h, pred_shape)

    ymin = tf.broadcast_to(tf.reshape(boxes[:, 0], [-1, 1, 1]), pred_shape)
    xmin = tf.broadcast_to(tf.reshape(boxes[:, 1], [-1, 1, 1]), pred_shape)
    ymax = tf.broadcast_to(tf.reshape(boxes[:, 2], [-1, 1, 1]), pred_shape)
    xmax = tf.broadcast_to(tf.reshape(boxes[:, 3], [-1, 1, 1]), pred_shape)

    mask_left = (cols >= xmin)
    mask_right = (cols <= xmax)
    mask_bottom = (rows >= ymin)
    mask_top = (rows <= ymax)

    crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right),
                                    tf.math.logical_and(mask_bottom, mask_top))
    crop_mask = tf.cast(crop_mask, tf.float32)
    # tf.print('crop', tf.shape(crop_mask))

    return pred * crop_mask


# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(anchors, loc_pred):
    # we use this variance also when we encode the offset
    variances = [0.1, 0.2]

    # convert anchor to center_form
    anchor_h = anchors[:, 2] - anchors[:, 0]
    anchor_w = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 1] + (anchor_w / 2)
    anchor_cy = anchors[:, 0] + (anchor_h / 2)
    tf.print("cx", tf.shape(anchor_cx))

    pred_cx, pred_cy, pred_w, pred_h = tf.unstack(loc_pred, axis=-1)

    new_cx = pred_cx * (anchor_w * variances[0]) + anchor_cx
    new_cy = pred_cy * (anchor_h * variances[0]) + anchor_cy
    new_w = tf.math.exp(pred_w * variances[1]) * anchor_w
    new_h = tf.math.exp(pred_h * variances[1]) * anchor_h

    ymin = new_cy - (new_h / 2)
    xmin = new_cx - (new_w / 2)
    ymax = new_cy + (new_h / 2)
    xmax = new_cx + (new_w / 2)

    decoded_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    tf.print(tf.shape(decoded_boxes))

    tf.print("anchor", tf.shape(anchors))
    tf.print("pred", tf.shape(loc_pred))
    return decoded_boxes
