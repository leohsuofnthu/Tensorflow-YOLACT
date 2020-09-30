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
# Restore CheckPoints
# -----------------------------------------------------------------------------------------------
lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

model = Yolact(input_size=550,
               fpn_channels=256,
               feature_map_size=[69, 35, 18, 9, 5],
               num_class=91,
               num_mask=32,
               aspect_ratio=[1, 0.5, 2],
               scales=[24, 48, 96, 192, 384])

ckpt_dir = "../checkpoints/"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(img_size=550,
                          feature_map_size=[69, 35, 18, 9, 5],
                          aspect_ratio=[1, 0.5, 2],
                          scale=[24, 48, 96, 192, 384])

# images for detection, new dataloader without repeating
# Todo: Figure out why batch size = 1 cause memory issue
valid_dataset = dataset_coco.prepare_dataloader(tfrecord_dir="../data/coco",
                                                batch_size=1,
                                                subset='val')
anchors = anchorobj.get_anchors()
tf.print(tf.shape(anchors))

# Add detection Layer after model
detection_layer = Detect(num_cls=91,
                         label_background=0,
                         top_k=200,
                         conf_threshold=0.05,
                         nms_threshold=0.8,
                         anchors=anchors)

# iteration for detection (5000 val images)
for image, labels in valid_dataset.take(1):
    # only try on 1 image
    output = model(image, training=False)
    detection = detection_layer(output)
    print(len(detection))
    # postprocessing
    cls, scores, bbox, masks = postprocess(detection, 550, 550, 0, "bilinear")
    tf.print("cls", tf.shape(cls))
    tf.print("scores", tf.shape(scores))
    tf.print("scores", scores)
    tf.print("bbox", tf.shape(bbox))
    tf.print("masks", tf.shape(masks))
    cls, scores, bbox, masks = cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
    # mAP_eval(detection) -> print table for batch
    # Todo evaluate mAP here

    # visualize the detection (un-transform the image)
    image = denormalize_image(image)
    image = np.squeeze(image.numpy()) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_bbox = labels['bbox'].numpy()
    gt_cls = labels['classes'].numpy()
    num_obj = labels['num_obj'].numpy()
    print(image.shape)

    # show the prediction box
    for idx in range(bbox.shape[0]):
        b = bbox[idx]
        print(cls[idx])
        print(COCO_LABEL_MAP.get(cls[idx]))
        if COCO_LABEL_MAP.get(cls[idx])-1 is None:
            continue
        class_id = COCO_LABEL_MAP.get(cls[idx])-1
        print(class_id)
        color_idx = (class_id) % len(COLORS)
        print(f"{class_id}, {COCO_CLASSES[class_id]}")

        # prepare the class text to display
        text_str = f"{COCO_CLASSES[class_id]}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        print(f"color {COLORS[color_idx]}")

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), COLORS[color_idx], 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), COLORS[color_idx], -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # show the label
    for idx in range(num_obj[0]):
        b = gt_bbox[0][idx]
        class_id = COCO_LABEL_MAP.get(gt_cls[0][idx]) - 1
        color_idx = (class_id * 5) % len(COLORS)
        print(f"{class_id}, {COCO_CLASSES[class_id]}")

        # prepare the class text to display
        text_str = f"{COCO_CLASSES[class_id]}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        print(f"color {COLORS[color_idx]}")

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (0, 0, 0), 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), (0, 0, 0), -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imshow("check", image)
    k = cv2.waitKey(0)


# Visualize Detection Results and calculate mAP / image with boxes, masks, scores, and class name
# -----------------------------------------------------------------------------------------------
