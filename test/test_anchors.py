import tensorflow as tf
from data import anchor

d = tf.Variable([0.1, 0.6, 0.7, 0.3])

anchors = tf.Variable([[2., 1., 5., 3.], [3., 6., 10., 8.], [5., 8., 12., 20.]])
gt_bbox = tf.Variable([[1., 6., 7., 10.], [1., 2., 7., 4.], [5., 6., 7., 9.]])
idx = tf.Variable([0, 1, 2])

map_loc = tf.map_fn(lambda x: gt_bbox[x], idx, dtype=tf.float32)
tf.print("mapping idx:\n", map_loc)


# to center form
def map_to_center_form(x):
    w = x[2] - x[0]
    h = x[3] - x[1]
    cx = x[0] + (w / 2)
    cy = x[1] + (h / 2)
    return tf.stack([cx, cy, w, h])


def map_to_point_form(x):
    xmin = x[0] - (x[2] / 2)
    ymin = x[1] - (x[3] / 2)
    xmax = x[0] + (x[2] / 2)
    ymax = x[1] + (x[3] / 2)
    return tf.stack([xmin, ymin, xmax, ymax])


def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = tf.math.log(x[2, 0] / x[2, 1])
    g_hat_h = tf.math.log(x[3, 0] / x[3, 1])
    return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])


center_anchors = tf.map_fn(lambda x: map_to_center_form(x), anchors)
center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)

tf.print("center_anchor:\n", center_anchors)
tf.print("center_gt:\n", center_gt)

concat = tf.stack([center_gt, center_anchors], axis=-1)
tf.print(concat[:, :, 0])
tf.print(concat[:, :, 1])

loc_target = tf.map_fn(lambda x: map_to_offset(x), concat)

tf.print("original anchors:\n", tf.map_fn(lambda x: map_to_point_form(x), center_anchors))
tf.print("offset:\n", loc_target)


# to center form


def test(x, pos, neg):
    if x < neg:
        return 0.
    elif x < pos:
        return -1.
    else:
        return 1.


p = tf.map_fn(lambda a: test(a, 0.5, 0.2), d)
tf.print(p)
