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
flags.DEFINE_integer('iter', 1,
                     'iteraitons')
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
        loc_loss, conf_loss, mask_loss, total_loss = loss_fn(output, labels, 91)
        logging.info("Total loss: %s..." %  total_loss)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss


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
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)

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
    iterations = FLAGS.iter
    for image, labels in train_dataset:
        if iterations > FLAGS.iter:
            break
        iterations += 1
        loc_loss, conf_loss, mask_loss = train_step(model, criterion, train_loss, optimizer, image, labels)
        loc.update_state(loc_loss)
        conf.update_state(conf_loss)
        mask.update_state(mask_loss)

        if iterations < FLAGS.iter and iterations % 100 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('Total loss', train_loss.result(), step=iterations)
                tf.summary.scalar('Loc loss', loc.result(), step=iterations)
                tf.summary.scalar('Conf loss', conf.result(), step=iterations)
                tf.summary.scalar('Mask loss', mask.result(), step=iterations)

            template = 'Epoch {}, Train Loss: {}, Loc Loss: {},  Conf Loss: {}, Mask Loss: {}'
            logging.info(template.format(iterations + 1,
                                         train_loss.result(),
                                         loc.result(),
                                         conf.result(),
                                         mask.result()))
            # reset the
            train_loss.reset_states()
            loc.reset_states()
            conf.reset_states()
            mask.reset_states()


if __name__ == '__main__':
    app.run(main)
