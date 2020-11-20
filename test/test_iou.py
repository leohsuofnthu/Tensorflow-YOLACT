import tensorflow as tf
from utils.utils import jaccard, intersection, mask_iou, bboxes_intersection

# ----------------------------------------------------------------------------------------------------------------------
# Test intersection, jaccard, and bboxes_intersection function
# gt [ymin, xmin, ymax, xmax]
"""
test_gt_bbox = tf.constant([[[2, 1, 5, 4]],
                            [[2, 2, 7, 5]],
                            [[4, 1, 6, 3]]], dtype=tf.float32)

# pred [ymin, xmin, ymax, xmax]
test_pred_bbox = tf.constant([[[1, 3, 3, 6]],
                              [[3, 3, 5, 5]],
                              [[1, 4, 3, 6]]], dtype=tf.float32)
"""
test_gt_bbox = tf.constant([[[2, 1, 5, 4]]], dtype=tf.float32)

# pred [ymin, xmin, ymax, xmax]
test_pred_bbox = tf.constant([[[1, 3, 3, 6]]], dtype=tf.float32)

tf.print(f"test gt", tf.shape(test_gt_bbox))
tf.print(f"test pred", tf.shape(test_pred_bbox))

inter = intersection(test_pred_bbox, test_gt_bbox)
tf.print(f"test intersection shape", tf.shape(inter))
tf.print(f"test intersection", inter)

jac = jaccard(test_pred_bbox, test_gt_bbox)
tf.print(f"test jaccard shape", tf.shape(jac))
tf.print(f"test jaccard", jac)
# ----------------------------------------------------------------------------------------------------------------------
# test bbox interaction for random crop
# Todo make it clear
test_ref_bbox = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
test_pred_bbox = tf.constant([[0.1, 0.3, 0.3, 0.6],
                              [0.3, 0.3, 0.5, 0.5],
                              [0.1, 0.4, 0.3, 0.6]], dtype=tf.float32)
bbox_iter = bboxes_intersection(test_ref_bbox, test_pred_bbox)
tf.print(f"test bbox inter shape", tf.shape(bbox_iter))
tf.print(f"test bbox inter", bbox_iter)
# ----------------------------------------------------------------------------------------------------------------------
# Test mask iou
test_gt_masks = tf.constant([
    [[1, 0, 0],
     [1, 0, 0],
     [1, 1, 1]],

    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],

    [[1, 0, 1],
     [1, 1, 1],
     [1, 0, 1]],
], dtype=tf.float32)

test_pred_masks = tf.constant([
    [[1, 1, 1],
     [1, 0, 0],
     [1, 0, 0]],

    [[1, 1, 1],
     [1, 0, 1],
     [0, 0, 0]]
], dtype=tf.float32)

tf.print(f"test gt mask", tf.shape(test_gt_masks))
tf.print(f"test pred mask", tf.shape(test_pred_masks))

m_iou = mask_iou(test_pred_masks, test_gt_masks)
tf.print(f"test mask iou shape", tf.shape(m_iou))
tf.print(f"test mask iou", m_iou)
