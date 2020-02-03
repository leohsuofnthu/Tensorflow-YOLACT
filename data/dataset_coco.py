"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import tensorflow as tf
from data import yolact_parser
from data import anchor
from yolact import Yolact
from loss.loss_yolact import YOLACTLoss


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)

def prepare_dataloader(tfrecord_dir, batch_size, subset="train"):
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "coco_%s.*" % subset))
    print(tf.shape(files))
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


train_dataloader = prepare_dataloader("./coco", 2, "train")
print(train_dataloader)

model = Yolact(input_size=550, fpn_channels=256, feature_map_size=[69, 35, 18, 9, 5], num_class=91, num_mask=4,
               aspect_ratio=[1, 0.5, 2], scales=[24, 48, 96, 192, 384])
model.build(input_shape=(4, 550, 550, 3))
model.summary()

criterion = YOLACTLoss()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
# Todo trying to train one epoch with loss calculated
for image, labels in train_dataloader:
    with tf.GradientTape() as tape:
        output = model(image)
        loss = criterion.loss(output, labels, 91)
        tf.print("loss:", loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))