import tensorflow as tf

from data import coco_dataset, anchor
from utils import learning_rate_schedule
from yolact import Yolact
from layers.detection import Detect

from eval import evaluate, print_maps

# set manual seed for easy debug
# -----------------------------------------------------------------------------------------------
tf.random.set_seed(1198)


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
valid_dataset = coco_dataset.prepare_dataloader(tfrecord_dir="../data/coco",
                                                batch_size=1,
                                                subset='val')
anchors = anchorobj.get_anchors()
tf.print(tf.shape(anchors))

# Add detection Layer after model
detection_layer = Detect(num_cls=91,
                         label_background=0,
                         top_k=200,
                         conf_threshold=0.05,
                         nms_threshold=0.5,
                         anchors=anchors)

# call evaluation(model, dataset)
# return calculated mAP
evaluate(model, detection_layer, valid_dataset, batch_size=1)


