import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data import dataset_coco, anchor
from utils import learning_rate_schedule, label_map
from yolact import Yolact
from layers.detection import Detect
from utils.utils import postprocess, denormalize_image
from utils.label_map import COCO_LABEL_MAP, COCO_CLASSES, COLORS

import cv2

# create dataset

# create model and load checkpoints

# call evaluation(model, dataset)

# return calculated mAP

# print the mAP in nice table
