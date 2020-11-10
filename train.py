import datetime
import contextlib
import tensorflow as tf
from tensorflow.keras.utils import Progbar

# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging

import yolact
from data import coco_dataset, anchor
from loss import loss_yolact
from utils import learning_rate_schedule

from eval import evaluate
from layers.detection import Detect

import config as cfg

tf.random.set_seed(1234)

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', './data/coco',
                    'directory of tfrecord')
flags.DEFINE_string('weights', './weights',
                    'path to store weights')
flags.DEFINE_integer('train_iter', 10000,
                     'iteraitons')
flags.DEFINE_integer('batch_size', 3,
                     'batch size')
flags.DEFINE_float('lr', 1e-3,
                   'learning rate')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
flags.DEFINE_float('weight_decay', 5 * 1e-4,
                   'weight_decay')
flags.DEFINE_float('print_interval', 10,
                   'number of iteration between printing loss')
flags.DEFINE_float('save_interval', 100,
                   'number of iteration between saving model(checkpoint)')
flags.DEFINE_float('valid_iter', 1000,
                   'number of iteration between saving validation weights')


@tf.function
def train_step(model,
               loss_fn,
               metrics,
               optimizer,
               image,
               labels):
    # training using tensorflow gradient tape
    with tf.GradientTape() as tape:
        output = model(image, training=True)
        # Todo how many category should we put in trianing
        loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(output, labels, 91)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss, seg_loss


@tf.function
def valid_step(model,
               loss_fn,
               metrics,
               image,
               labels):
    # Todo Calculate mAP here for evaluation
    output = model(image, training=False)
    loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(output, labels, 91)
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss, seg_loss


def main(argv):
    # set up Grappler for graph optimization
    # Ref: https://www.tensorflow.org/guide/graph_optimization
    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the dataloader from: %s..." % FLAGS.tfrecord_dir)
    train_dataset = coco_dataset.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    batch_size=FLAGS.batch_size,
                                                    subset='train',
                                                    **cfg.parser_params)

    valid_dataset = coco_dataset.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    batch_size=1,
                                                    subset='val',
                                                    **cfg.parser_params)
    # -----------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Creating the model instance of YOLACT")
    model = yolact.Yolact(**cfg.model_parmas)

    # Need default anchor
    anchorobj = anchor.Anchor(**cfg.anchor_params)
    anchors = anchorobj.get_anchors()

    # Add detection Layer after model
    detection_layer = Detect(**cfg.detection_params)

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.bias))

    # -----------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics, learning rate schedule
    lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4,
                                                                     initial_lr=FLAGS.lr)
    logging.info("Initiate the Optimizer and Loss function...")
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=FLAGS.momentum)
    criterion = loss_yolact.YOLACTLoss()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
    seg = tf.keras.metrics.Mean('seg_loss', dtype=tf.float32)
    v_loc = tf.keras.metrics.Mean('vloc_loss', dtype=tf.float32)
    v_conf = tf.keras.metrics.Mean('vconf_loss', dtype=tf.float32)
    v_mask = tf.keras.metrics.Mean('vmask_loss', dtype=tf.float32)
    v_seg = tf.keras.metrics.Mean('vseg_loss', dtype=tf.float32)

    # -----------------------------------------------------------------

    # Setup the TensorBoard for better visualization
    # Ref: https://www.tensorflow.org/tensorboard/get_started
    logging.info("Setup the TensorBoard...")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # -----------------------------------------------------------------
    # Start the Training and Validation Process
    logging.info("Start the training process...")

    # setup checkpoints manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=5
    )
    # restore from latest checkpoint and iteration
    status = checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    best_val = 1e10
    iterations = checkpoint.step.numpy()

    for image, labels in train_dataset:
        # check iteration and change the learning rate
        if iterations > FLAGS.train_iter:
            break

        checkpoint.step.assign_add(1)
        iterations += 1
        with options({'constant_folding': True,
                      'layout_optimize': True,
                      'loop_optimization': True,
                      'arithmetic_optimization': True,
                      'remapping': True}):
            loc_loss, conf_loss, mask_loss, seg_loss = train_step(model, criterion, train_loss, optimizer, image,
                                                                  labels)
        loc.update_state(loc_loss)
        conf.update_state(conf_loss)
        mask.update_state(mask_loss)
        seg.update_state(seg_loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('Total loss', train_loss.result(), step=iterations)
            tf.summary.scalar('Loc loss', loc.result(), step=iterations)
            tf.summary.scalar('Conf loss', conf.result(), step=iterations)
            tf.summary.scalar('Mask loss', mask.result(), step=iterations)
            tf.summary.scalar('Seg loss', seg.result(), step=iterations)

        if iterations and iterations % FLAGS.print_interval == 0:
            logging.info("Iteration {}, LR: {}, Total Loss: {}, B: {},  C: {}, M: {}, S:{} ".format(
                iterations,
                optimizer._decayed_lr(var_dtype=tf.float32),
                train_loss.result(), loc.result(),
                conf.result(),
                mask.result(),
                seg.result()
            ))

        if iterations and iterations % FLAGS.save_interval == 0:
            # save checkpoint
            save_path = manager.save()
            logging.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            # validation and print mAP table
            # evaluate(model, detection_layer, valid_dataset)
            valid_iter = 0
            for valid_image, valid_labels in valid_dataset:
                if valid_iter > FLAGS.valid_iter:
                    break
                # calculate validation loss
                with options({'constant_folding': True,
                              'layout_optimize': True,
                              'loop_optimization': True,
                              'arithmetic_optimization': True,
                              'remapping': True}):
                    # Todo accumulate mAP calculation, call evaluate(model, dataset) directly and print the mAP
                    # get the detections, saving in objects
                    valid_loc_loss, valid_conf_loss, valid_mask_loss, valid_seg_loss = valid_step(model,
                                                                                                  criterion,
                                                                                                  valid_loss,
                                                                                                  valid_image,
                                                                                                  valid_labels)
                v_loc.update_state(valid_loc_loss)
                v_conf.update_state(valid_conf_loss)
                v_mask.update_state(valid_mask_loss)
                v_seg.update_state(valid_seg_loss)
                valid_iter += 1

            with test_summary_writer.as_default():
                tf.summary.scalar('V Total loss', valid_loss.result(), step=iterations)
                tf.summary.scalar('V Loc loss', v_loc.result(), step=iterations)
                tf.summary.scalar('V Conf loss', v_conf.result(), step=iterations)
                tf.summary.scalar('V Mask loss', v_mask.result(), step=iterations)
                tf.summary.scalar('V Seg loss', v_seg.result(), step=iterations)

            train_template = 'Iteration {}, Train Loss: {}, Loc Loss: {},  Conf Loss: {}, Mask Loss: {}, Seg Loss: {}'
            valid_template = 'Iteration {}, Valid Loss: {}, V Loc Loss: {},  V Conf Loss: {}, V Mask Loss: {}, ' \
                             'Seg Loss: {} '
            logging.info(train_template.format(iterations + 1,
                                               train_loss.result(),
                                               loc.result(),
                                               conf.result(),
                                               mask.result(),
                                               seg.result()))
            logging.info(valid_template.format(iterations + 1,
                                               valid_loss.result(),
                                               v_loc.result(),
                                               v_conf.result(),
                                               v_mask.result(),
                                               v_seg.result()))

            # Todo save the best mAP
            """
            if valid_loss.result() < best_val:
                # Saving the weights:
                best_val = valid_loss.result()
                model.save_weights('./weights/weights_' + str(valid_loss.result().numpy()) + '.h5')
            """
            # reset the metrics
            train_loss.reset_states()
            loc.reset_states()
            conf.reset_states()
            mask.reset_states()
            seg.reset_states()

            valid_loss.reset_states()
            v_loc.reset_states()
            v_conf.reset_states()
            v_mask.reset_states()
            v_seg.reset_states()


if __name__ == '__main__':
    app.run(main)
