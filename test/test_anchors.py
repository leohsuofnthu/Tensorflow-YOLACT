import numpy as np
import tensorflow as tf
import config as cfg
from data.anchor import Anchor


test_bbox = tf.convert_to_tensor((np.array([[204.044, 253.8351, 487.8226, 427.06363],
                                            [0, 140.01741, 550, 290.21936],
                                            [40.005028, 117.37102, 255.7913, 205.13097],
                                            [263.31314, 67.0434, 514.04736, 124.48139],
                                            [0, 503.79834, 487.0279, 550]])), dtype=tf.float32)

test_labels = tf.convert_to_tensor((np.array([[1],
                                              [2],
                                              [3],
                                              [4],
                                              [5]])), dtype=tf.float32)

anchorobj = Anchor(**cfg.anchor_params)
print(anchorobj.get_anchors())

target_cls, target_loc, max_id_for_anchors, match_positiveness = anchorobj.matching(threshold_pos=0.5,
                                                                                    threshold_neg=0.4,
                                                                                    gt_bbox=test_bbox,
                                                                                    gt_labels=test_labels)

print(target_loc)
