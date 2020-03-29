import os
import tensorflow as tf
import matplotlib.pyplot as plt

from data import dataset_coco, anchor
from utils import learning_rate_schedule, label_map
from yolact import Yolact
from layers.detection import Detect
from utils.utils import postprocess, denormalize_image

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

detection_layer = Detect(91, 0, 200, 0.5, 0.5, anchors)

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
    tf.print("bbox", tf.shape(bbox))
    tf.print("masks", tf.shape(masks))
    cls, scores, bbox, masks = cls.numpy(), scores.numpy(), bbox.numpy(), masks.numpy()
    # visualize the detection (un-transform the image)
    image = denormalize_image(image)
    image = tf.squeeze(image).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_bbox = labels['bbox'].numpy()
    gt_cls = labels['classes'].numpy()
    num_obj = labels['num_obj'].numpy()
    print(image.shape)
    for idx in range(bbox.shape[0]):
        b = bbox[idx]
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (255, 0, 0), 2)
        cv2.putText(image, label_map.category_map[cls[idx]+1], (int(b[1]), int(b[0]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (36, 255, 12), 2)

    for idx in range(num_obj[0]):
        b = gt_bbox[0][idx]
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (0, 0, 255), 2)
        cv2.putText(image, label_map.category_map[gt_cls[0][idx]], (int(b[1]), int(b[0]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (36, 18, 12), 2)

    cv2.imshow("check", image)
    k = cv2.waitKey(0)


# Visualize Detection Results and calculate mAP / image with boxes, masks, scores, and class name
# -----------------------------------------------------------------------------------------------
