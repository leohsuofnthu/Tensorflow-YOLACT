import os

import cv2
import numpy as np
import tensorflow as tf

from config import PASCAL_CLASSES, COLORS, get_params, ROOT_DIR
from data.coco_dataset import ObjectDetectionDataset
from utils import learning_rate_schedule
from utils.utils import postprocess, denormalize_image
from yolact import Yolact

# Todo Add your custom dataset
NAME_OF_DATASET = "pascal"
CLASS_NAMES = PASCAL_CLASSES

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
# Restore CheckPoints
# Choose the Optimizor, Loss Function, and Metrics, learning rate schedule
lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(**lrs_schedule_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")

# -----------------------------------------------------------------------------------------------
# Load Validation Images and do Detection

# iteration for detection (5000 val images)
for image, labels in valid_dataset.take(1):
    # only try on 1 image
    output = model(image, training=False)
    detection = model.detect(output)
    # postprocessing
    cls, scores, bbox, masks = postprocess(detection, tf.shape(image)[0], tf.shape(image)[1], 0, "bilinear")
    cls, scores, bbox, masks = cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
    # visualize the detection (un-transform the image)
    image = denormalize_image(image)
    image = np.squeeze(image.numpy()) * 255
    image = image.astype(np.uint8)
    gt_bbox = labels['bbox'].numpy()
    gt_cls = labels['classes'].numpy()
    num_obj = labels['num_obj'].numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    final_m = np.zeros_like(masks[0][:, :, None])
    # show the prediction box
    for idx in range(bbox.shape[0]):
        b = bbox[idx]
        m = masks[idx][:, :, None]
        class_id = cls[idx]
        color_idx = class_id % len(COLORS)
        score = '%.2f' % round(scores[idx], 2)
        # prepare the class text to display
        text_str = f"{CLASS_NAMES[class_id]} {score}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        color = COLORS[color_idx]

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), COLORS[color_idx], 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), COLORS[color_idx], -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # create mask
        final_m = final_m + np.concatenate((m * color[0], m * color[1], m * color[2]), axis=-1)

    final_m = final_m.astype('uint8')
    dst = np.zeros_like(image).astype('uint8')
    final_m = cv2.resize(final_m, dsize=(image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    cv2.addWeighted(final_m, 0.3, image, 0.7, 0, dst)
    cv2.imshow("check", dst)
    k = cv2.waitKey(0)
