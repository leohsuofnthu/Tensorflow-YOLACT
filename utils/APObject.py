"""
REF: https://github.com/dbolya/yolact/blob/master/eval.py
"""
import json


class APObject:
    """
    Object to store mAP related information for 1 IOU threshhold and 1 class
    Ex: class "cat" 's mAP at threshold 0.5 is stored into a APObject
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score, is_true):
        ...

    def add_gt_positive(self, num_positives):
        ...

    def is_empty(self):
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self):
        if self.num_gt_positives == 0:
            return 0



class Detections:
    """
    Collection of detected information (include bbox and mask)
    """

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_box(self):
        ...

    def add_mask(self):
        ...

    def to_json(self):
        """
        dump to json file for benchmark use, like coco-test result
        """
        ...
