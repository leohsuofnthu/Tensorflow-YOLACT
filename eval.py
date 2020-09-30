import os

import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from utils.APObject import APObject, Detections
from utils.utils import jaccard, mask_iou

iou_thresholds = [x / 100 for x in range(50, 100, 5)]


def _bbox_iou(bbox1, bbox2):
    ret = jaccard(bbox1, bbox2)
    return ret


def _mask_iou(mask1, mask2):
    ret = mask_iou(mask1, mask2)
    return ret


def eval_image():
    ...


def eval_images():
    ...


def eval_video():
    ...


def evaluate():
    ...


def main(argv):
    pass


if __name__ == '__main__':
    app.run(main)
