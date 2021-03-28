import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import PASCAL_CLASSES, COCO_CLASSES, COCO_LABEL_MAP, COLORS, get_params, ROOT_DIR
from data.coco_dataset import ObjectDetectionDataset
from utils.utils import denormalize_image
from yolact import Yolact

# Todo Add your custom dataset
NAME_OF_DATASET = "coco"
CLASS_NAMES = COCO_CLASSES
LABEL_REMAP = COCO_LABEL_MAP

# -----------------------------------------------------------------------------------------------
# create model and dataloader
train_iter, input_size, num_cls, lrs_schedule_params, loss_params, parser_params, model_params = get_params(
    NAME_OF_DATASET)
model = Yolact(**model_params)
dateset = ObjectDetectionDataset(dataset_name=NAME_OF_DATASET,
                                 tfrecord_dir=os.path.join(ROOT_DIR, "data", NAME_OF_DATASET),
                                 anchor_instance=model.anchor_instance,
                                 **parser_params)
train_dataset = dateset.get_dataloader(subset='train', batch_size=1)
valid_dataset = dateset.get_dataloader(subset='val', batch_size=1)
# -----------------------------------------------------------------------------------------------
for image, labels in train_dataset.take(1):
    image = denormalize_image(image)
    image = np.squeeze(image.numpy()) * 255
    image = image.astype(np.uint8)
    ori = np.squeeze(labels['ori'].numpy())
    plt.imshow(ori)
    plt.show()
    bbox = labels['bbox'].numpy()
    cls = labels['classes'].numpy()
    mask = labels['mask_target'].numpy()
    num_obj = labels['num_obj'].numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    final_m = np.zeros_like(mask[0][0][:, :, None])
    for idx in range(num_obj[0]):
        # get the bbox, class_name, and random color
        b = bbox[0][idx]
        m = mask[0][idx][:, :, None]
        if LABEL_REMAP:
            class_id = LABEL_REMAP.get(cls[0][idx]) - 1
        class_id = cls[0][idx] - 1
        color_idx = (class_id * 5) % len(COLORS)

        # prepare the class text to display
        text_str = f"{CLASS_NAMES[class_id]}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.4
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        color = COLORS[color_idx]

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), color, 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), color, -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # create mask
        final_m = final_m + np.concatenate((m * color[0], m * color[1], m * color[2]), axis=-1)
        plt.imshow(final_m)
    plt.show()

    dst = np.zeros_like(image).astype('uint8')
    final_m = tf.image.resize(final_m, [image.shape[0], image.shape[1]], method=tf.image.ResizeMethod.BILINEAR)
    final_m = (final_m + 0.5).numpy().astype('uint8')
    cv2.addWeighted(final_m, 0.3, image, 0.7, 0, dst)
    cv2.imshow("check", dst)
    k = cv2.waitKey(0)
# ---------------------------------------------------------------------------------------------------------------
# for visualizing a crowd example
"""
# visualize the first crowd training sample
for image, labels in train_dataloader:
    if labels['num_crowd'] > 0:
        image = denormalize_image(image)
        image = np.squeeze(image.numpy()) * 255
        image = image.astype(np.uint8)
        ori = labels['ori']
        ori = np.squeeze(labels['ori'].numpy())
        plt.imshow(ori)
        plt.show()
        bbox = labels['bbox'].numpy()
        cls = labels['classes'].numpy()
        mask = labels['mask_target'].numpy()
        num_obj = labels['num_obj'].numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_m = np.zeros_like(mask[0][0][:, :, None])
        idx = num_obj[0]-1
        # get the bbox, class_name, and random color
        b = bbox[0][idx]
        m = mask[0][idx][:, :, None]
        class_id = COCO_LABEL_MAP.get(cls[0][idx]) - 1
        color_idx = (class_id * 5) % len(COLORS)

        # prepare the class text to display
        text_str = f"{COCO_CLASSES[class_id]}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        color = COLORS[color_idx]

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), color, 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), color, -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # create mask
        final_m = final_m + np.concatenate((m * color[0], m * color[1], m * color[2]), axis=-1)

        final_m = final_m.astype('uint8')
        dst = np.zeros_like(image).astype('uint8')
        final_m = cv2.resize(final_m, dsize=(image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
        cv2.addWeighted(final_m, 0.3, image, 0.7, 0, dst)
        cv2.imshow("check", dst)
        k = cv2.waitKey(0)

    else:
        continue
"""
