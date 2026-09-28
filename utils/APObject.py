"""
Adapted from https://github.com/dbolya/yolact/blob/master/eval.py
"""
import json

import numpy as np

try:
    import pycocotools.mask as mask_util
except ImportError:  # optional; only needed for COCO JSON export
    mask_util = None


class Detections:
    """Collection of detected information (bbox and mask) for COCO JSON export."""

    def __init__(self, label_map=None):
        self.bbox_data = []
        self.mask_data = []
        # maps model class index (0-based, without background) -> COCO category id
        self.label_map = label_map or {}

    def _coco_category_id(self, category_id):
        """category_id is 0-based class index from the model (background excluded)."""
        # COCO_LABEL_MAP is {coco_id: model_id} with model_id in 1..80
        model_id = int(category_id) + 1
        for coco_id, mapped in self.label_map.items():
            if mapped == model_id:
                return int(coco_id)
        return model_id

    def add_box(self, image_id, category_id, bbox, score):
        """bbox: (x1, y1, x2, y2) in absolute image coordinates."""
        x1, y1, x2, y2 = bbox
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]
        coco_bbox = [round(float(v) * 10) / 10 for v in coco_bbox]
        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': self._coco_category_id(category_id),
            'bbox': coco_bbox,
            'score': float(score),
        })

    def add_mask(self, image_id, category_id, segmentation, score):
        """segmentation: full-image mask [H, W]."""
        if mask_util is None:
            raise ImportError(
                'pycocotools is required for mask JSON export. '
                'Install it with: pip install pycocotools'
            )
        rle = mask_util.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': self._coco_category_id(category_id),
            'segmentation': rle,
            'score': float(score),
        })

    def to_json(self, bbox_path=None, mask_path=None):
        """Write COCO-style detection JSON files."""
        if bbox_path:
            with open(bbox_path, 'w', encoding='utf-8') as f:
                json.dump(self.bbox_data, f)
        if mask_path:
            with open(mask_path, 'w', encoding='utf-8') as f:
                json.dump(self.mask_data, f)


class APObject:
    """
    Object to store mAP related information for 1 IoU threshold and 1 class.
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

        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        true_positive = 0
        false_positive = 0

        for datapoint in self.data_points:
            if datapoint[1]:
                true_positive += 1
            else:
                false_positive += 1

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / self.num_gt_positives
            precisions.append(precision)
            recalls.append(recall)

        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        y_range = [0] * 101
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        return sum(y_range) / len(y_range)
