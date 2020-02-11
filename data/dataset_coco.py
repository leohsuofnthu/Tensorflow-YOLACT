"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import datetime
import tensorflow as tf
from data import yolact_parser
from data import anchor
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import label_map
from loss.loss_yolact import YOLACTLoss
from yolact import Yolact


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)

def prepare_dataloader(tfrecord_dir, batch_size, subset="train"):
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    # print(tf.shape(files))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))  # wtf?
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=1024)
    anchorobj = anchor.Anchor(img_size=550,
                              feature_map_size=[69, 35, 18, 9, 5],
                              aspect_ratio=[1, 0.5, 2],
                              scale=[24, 48, 96, 192, 384])
    parser = yolact_parser.Parser(output_size=550,
                                  anchor_instance=anchorobj,
                                  match_threshold=0.5,
                                  unmatched_threshold=0.5,
                                  mode=subset)

    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    return dataset


"""
train_dataloader = prepare_dataloader("./coco", 8, "train")
print(train_dataloader)
# visualize the training sample
# Sets up a timestamped log directory.
logdir = "../logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)
count = 0
for image, label in train_dataloader:
    image = np.squeeze(image.numpy())
    bbox = labels['bbox'].numpy()
    cls = labels['classes'].numpy()
    mask = labels['mask_target'].numpy()
    for idx in range(5):
        b = bbox[0][idx]
        cv2.rectangle(image, (b[1], b[0]), (b[3],b[2]), (255,0,0), 2)
        cv2.putText(image, label_map.category_map[cls[0][idx]], (int(b[1]), int(b[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
        plt.figure()
        plt.imshow(mask[0][idx])

    plt.show()
    cv2.imshow("check", image)
    k = cv2.waitKey(0)
    print(cls)
    break
"""
