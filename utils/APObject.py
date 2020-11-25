"""
Adapted from https://github.com/dbolya/yolact/blob/master/eval.py
"""
import numpy as np


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

        # compute points in precision-recall curve
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

        # compute AP, some details needed
        # smooth the curve
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation
        # with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)
