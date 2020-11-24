import tensorflow as tf

from data import coco_dataset, anchor
from utils import learning_rate_schedule
from yolact import Yolact
from layers.detection import Detect

from eval import evaluate, print_maps
import config as cfg
# set manual seed for easy debug
# -----------------------------------------------------------------------------------------------
tf.random.set_seed(1198)


# Restore CheckPoints
# -----------------------------------------------------------------------------------------------
lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

model = Yolact(**cfg.model_parmas)

ckpt_dir = "../checkpoints/"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")

# Load Validation Images and do Detection
# -----------------------------------------------------------------------------------------------
# Need default anchor
anchorobj = anchor.Anchor(**cfg.anchor_params)

# images for detection, new dataloader without repeating
# Todo: Figure out why batch size = 1 cause memory issue
valid_dataset = coco_dataset.prepare_dataloader(tfrecord_dir="../data/coco",
                                                batch_size=1,
                                                subset='val',
                                                **cfg.parser_params)
anchors = anchorobj.get_anchors()
tf.print(tf.shape(anchors))

# Add detection Layer after model
detection_layer = Detect(**cfg.detection_params)

# call evaluation(model, dataset)
# return calculated mAP
evaluate(model, detection_layer, valid_dataset, batch_size=1)


