"""
Adapted from https://github.com/dbolya/yolact/blob/master/eval.py
"""
import json


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
        dump to json file for benchmark use, for coco-test dev benchmark
        """
        ...


class APObject:
    """
    Object to store mAP related information for 1 IOU threshhold (0.5 ~ 0.95) and 1 class (80)
    Ex: class "cat" 's mAP at threshold 0.5 is stored into a APObject
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score, is_true):
        self.data_points.append((score, is_true))

    def add_gt_positive(self, num_positives):
        self.num_gt_positives += num_positives

    def is_empty(self):
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self):
        if self.num_gt_positives == 0:
            return 0

        # Sort by score in descending order
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        true_positive = 0
        false_positive = 0

        # compute the precision-recall curve
        # X-axis: recalls Y-axis: precisions
        for datapoint in self.data_points:
            # check if the detection is true or false positive
            if datapoint[1]:
                true_positive += 1
            else:
                false_positive += 1

            # calculate precision and recall
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)
