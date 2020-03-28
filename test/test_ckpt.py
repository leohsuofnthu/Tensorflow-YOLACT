import os
import tensorflow as tf

from data import dataset_coco, anchor
from utils import learning_rate_schedule
from yolact import Yolact
from layers.detection import Detect

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
valid_dataset = dataset_coco.prepare_dataloader(tfrecord_dir="../data/coco",
                                                batch_size=2,
                                                subset='val')
anchors = anchorobj.get_anchors()
tf.print(tf.shape(anchors))

detection_layer = Detect(91, 0, 200, 0.5, 0.5, anchors)

# iteration for detection (5000 val images)
for image, labels in valid_dataset:
    output = model(image, training=False)
    detection_layer(output)
    # visualize the detection

    break

# Visualize Detection Results
# -----------------------------------------------------------------------------------------------
