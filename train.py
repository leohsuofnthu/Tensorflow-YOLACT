import time
import datetime

# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from data import dataset_coco
from loss import loss_yolact
import yolact

from tensorflow.keras.mixed_precision import experimental as mixed_precision

import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import label_map

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', './data/coco',
                    'directory of tfrecord')
flags.DEFINE_string('weights', './weights',
                    'path to store weights')
flags.DEFINE_integer('train_iter', 100,
                     'iteraitons')
flags.DEFINE_integer('batch_size', 2,
                     'batch size')
flags.DEFINE_float('lr', 1e-4,
                   'learning rate')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
flags.DEFINE_float('weight_decay', 5 * 1e-4,
                   'weight_decay')
flags.DEFINE_float('save_interval', 10,
                   'number of iteration between saving model')
flags.DEFINE_float('valid_iter', 50,
                   'number of iteration between saving model')

logging.set_verbosity(logging.INFO)


def train_step(model,
               loss_fn,
               metrics,
               optimizer,
               image,
               labels):
    # training using tensorflow gradient tape
    with tf.GradientTape() as tape:
        output = model(image)
        loc_loss, conf_loss, mask_loss, total_loss = loss_fn(output, labels, 91)
        logging.info("Total loss: %s..." % total_loss)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss


def valid_step(model,
               loss_fn,
               metrics,
               image,
               labels):
    output = model(image)
    loc_loss, conf_loss, mask_loss, total_loss = loss_fn(output, labels, 91)
    logging.info("Total loss: %s..." % total_loss)
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss


def main(argv):
    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the dataloader from: %s..." % FLAGS.tfrecord_dir)
    train_dataset = dataset_coco.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    batch_size=FLAGS.batch_size,
                                                    subset='train')

    valid_dataset = dataset_coco.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    batch_size=FLAGS.batch_size,
                                                    subset='val')
    # -----------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Creating the model instance of YOLACT")
    model = yolact.Yolact(input_size=550,
                          fpn_channels=16,
                          feature_map_size=[69, 35, 18, 9, 5],
                          num_class=91,
                          num_mask=4,
                          aspect_ratio=[1, 0.5, 2],
                          scales=[24, 48, 96, 192, 384])
    # model.build(input_shape=(4, 550, 550, 3))
    # model.summary()

    # initialization of the parameters

    # -----------------------------------------------------------------

    # Choose the Optimizor, Loss Function, and Metrics
    logging.info("Initiate the Optimizer and Loss function...")
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr, momentum=FLAGS.momentum, decay=FLAGS.weight_decay)
    criterion = loss_yolact.YOLACTLoss()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
    v_loc = tf.keras.metrics.Mean('vloc_loss', dtype=tf.float32)
    v_conf = tf.keras.metrics.Mean('vconf_loss', dtype=tf.float32)
    v_mask = tf.keras.metrics.Mean('vmask_loss', dtype=tf.float32)

    # -----------------------------------------------------------------

    # Setup the TensorBoard for better visualization
    # Ref: https://www.tensorflow.org/tensorboard/get_started
    logging.info("Setup the TensorBoard...")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # -----------------------------------------------------------------

    # Start the Training and Validation Process
    logging.info("Start the training process...")
    iterations = 0
    # Freeze the BN layers in pre-trained backbone
    model.set_bn('train')
    t0 = time.time()
    for image, labels in train_dataset:
        """
        i = np.squeeze(image.numpy())
        bbox = labels['bbox'].numpy()
        cls = labels['classes'].numpy()
        m = labels['mask_target'].numpy()
        for idx in range(2):
            b = bbox[0][idx]
            print(b)
            cv2.rectangle(i, (b[1], b[0]), (b[3], b[2]), (255, 0, 0), 2)
            cv2.putText(i, label_map.category_map[cls[0][idx]], (int(b[1]), int(b[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
            plt.figure()
            plt.imshow(m[0][idx])
        cv2.imshow("check", i)
        k = cv2.waitKey(0)
        """
        # check iteration and change the learning rate
        if iterations > FLAGS.train_iter:
            break

        iterations += 1
        loc_loss, conf_loss, mask_loss = train_step(model, criterion, train_loss, optimizer, image, labels)
        loc.update_state(loc_loss)
        conf.update_state(conf_loss)
        mask.update_state(mask_loss)

        with train_summary_writer.as_default():
            tf.summary.scalar('Total loss', train_loss.result(), step=iterations)
            tf.summary.scalar('Loc loss', loc.result(), step=iterations)
            tf.summary.scalar('Conf loss', conf.result(), step=iterations)
            tf.summary.scalar('Mask loss', mask.result(), step=iterations)
        if iterations < FLAGS.train_iter and iterations % FLAGS.save_interval == 0:
            # validation
            valid_iter = 0
            for valid_image, valid_labels in valid_dataset:
                if valid_iter > FLAGS.valid_iter:
                    break
                # calculate validation loss
                valid_loc_loss, valid_conf_loss, valid_mask_loss = valid_step(model, criterion, valid_loss,
                                                                              valid_image, valid_labels)
                v_loc.update_state(valid_loc_loss)
                v_conf.update_state(valid_conf_loss)
                v_mask.update_state(valid_mask_loss)
                valid_iter += 1

            with test_summary_writer.as_default():
                tf.summary.scalar('V Total loss', valid_loss.result(), step=iterations)
                tf.summary.scalar('V Loc loss', v_loc.result(), step=iterations)
                tf.summary.scalar('V Conf loss', v_conf.result(), step=iterations)
                tf.summary.scalar('V Mask loss', v_mask.result(), step=iterations)

            train_template = 'Iteration {}, Train Loss: {}, Loc Loss: {},  Conf Loss: {}, Mask Loss: {}'
            valid_template = 'Iteration {}, Valid Loss: {}, V Loc Loss: {},  V Conf Loss: {}, V Mask Loss: {}'
            logging.info(train_template.format(iterations + 1,
                                               train_loss.result(),
                                               loc.result(),
                                               conf.result(),
                                               mask.result()))
            logging.info(valid_template.format(iterations + 1,
                                               valid_loss.result(),
                                               v_loc.result(),
                                               v_conf.result(),
                                               v_mask.result()))
            # reset the metrics
            train_loss.reset_states()
            loc.reset_states()
            conf.reset_states()
            mask.reset_states()

            valid_loss.reset_states()
            v_loc.reset_states()
            v_conf.reset_states()
            v_mask.reset_states()
            t1 = time.time()
            logging.info("Training interval: %s second" % (t1 - t0))
            t0 = time.time()


if __name__ == '__main__':
    app.run(main)
