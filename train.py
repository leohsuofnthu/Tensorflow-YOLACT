import datetime

# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from data import dataset_coco
from loss import loss_yolact
import yolact

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', './data/coco',
                    'directory of tfrecord')
flags.DEFINE_integer('epochs', 100,
                     'epochs')
flags.DEFINE_integer('batch_size', 1,
                     'batch size')
flags.DEFINE_float('lr', 1e-3,
                   'learning rate')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
flags.DEFINE_float('weight_decay', 5 * 1e-4,
                   'weight_decay')

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
        loss = loss_fn(output, labels, 91)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(loss)


def valid_step():
    pass


def main(argv):
    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the dataloader from: %s..." % FLAGS.tfrecord_dir)
    train_dataset = dataset_coco.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    batch_size=FLAGS.batch_size,
                                                    subset='train')

    """
    valid_dataset = dataloader.prepare_data_loader(df_path=os.path.join(FLAGS.dataframe),
                                                   crop_size=FLAGS.crop_size,
                                                   batch_size=FLAGS.batch_size,
                                                   subset='valid')
    """
    # -----------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Creating the model instance of YOLACT")
    model = yolact.Yolact(input_size=550,
                          fpn_channels=256,
                          feature_map_size=[69, 35, 18, 9, 5],
                          num_class=91,
                          num_mask=4,
                          aspect_ratio=[1, 0.5, 2],
                          scales=[24, 48, 96, 192, 384])
    model.build(input_shape=(4, 550, 550, 3))
    model.summary()
    # -----------------------------------------------------------------

    # Choose the Optimizor, Loss Function, and Metrics
    logging.info("Initiate the Optimizer and Loss function...")
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr, momentum=FLAGS.momentum)
    criterion = loss_yolact.YOLACTLoss()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

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
    for epoch in range(FLAGS.epochs):
        for image, labels in train_dataset:
            train_step(model, criterion, train_loss, optimizer, image, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('train loss', train_loss.result(), step=epoch)


if __name__ == '__main__':
    app.run(main)
