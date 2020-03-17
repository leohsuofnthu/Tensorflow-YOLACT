import datetime

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.dataset_coco import prepare_dataloader
from loss import loss_yolact
from utils import label_map
from yolact import Yolact

"""
model = Yolact(input_size=550, fpn_channels=256, feature_map_size=[69, 35, 18, 9, 5], num_class=91, num_mask=32,
               aspect_ratio=[1, 0.5, 2], scales=[24, 48, 96, 192, 384])
model.build(input_shape=(8, 550, 550, 3))

model.summary()

loss_fn = loss_yolact.YOLACTLoss()

train_dataloader = prepare_dataloader("../data/coco", 1, "train")
print(train_dataloader)
# visualize the training sample
# Sets up a timestamped log directory.
logdir = "../logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)
count = 0
for image, labels in train_dataloader:
    output = model(image)
    loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(output, labels, 91)
    print(loc_loss, conf_loss, mask_loss, seg_loss, total_loss)
    image = np.squeeze(image.numpy())
    bbox = labels['bbox'].numpy()
    cls = labels['classes'].numpy()
    mask = labels['mask_target'].numpy()

    file = h5py.File("path", "w")
    file.create_dataset("bbox", np.shape(bbox), data=bbox)
    file.create_dataset("cls", np.shape(cls), data=cls)
    file.create_dataset("mask", np.shape(mask), data=mask)
    print("Successfully saved -> ", "label.h5")
    file.close()

    num_obj = labels['num_obj'].numpy()
    plt.figure()
    plt.imshow(image)
    for idx in range(num_obj[0]):
        b = bbox[0][idx]
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (255, 0, 0), 2)
        cv2.putText(image, label_map.category_map[cls[0][idx]], (int(b[1]), int(b[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (36, 255, 12), 2)
        plt.figure()
        plt.imshow(mask[0][idx])
    cv2.imshow("check", image)
    k = cv2.waitKey(0)
    plt.show()
    print(cls)
    break
"""
import time

train_dataloader = prepare_dataloader("../data/coco", 8, "train")

count = 1
t0 = time.time()
for img, label in train_dataloader:
    count += 1
    if count > 10:
        break
t1 = time.time()
print("10 fetch: %s" % (t1 - t0))
