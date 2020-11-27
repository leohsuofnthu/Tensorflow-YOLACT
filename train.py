import contextlib
import datetime
import os

import tensorflow as tf
# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from config import RANDOM_SEED, get_params, MIXPRECISION
from data.coco_dataset import ObjectDetectionDataset
from eval import evaluate
from loss import loss_yolact
from utils import learning_rate_schedule
from yolact import Yolact

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'coco',
                    'name of dataset')
flags.DEFINE_string('tfrecord_dir', 'data',
                    'directory of tfrecord')
flags.DEFINE_string('weights', 'weights',
                    'path to store weights')
flags.DEFINE_integer('batch_size', 3,
                     'batch size')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
flags.DEFINE_float('weight_decay', 5 * 1e-4,
                   'weight_decay')
flags.DEFINE_float('print_interval', 10,
                   'number of iteration between printing loss')
flags.DEFINE_float('save_interval', 100,
                   'number of iteration between saving model(checkpoint)')


@tf.function
def train_step(model,
               loss_fn,
               metrics,
               optimizer,
               image,
               labels,
               num_cls):
    # training using tensorflow gradient tape
    with tf.GradientTape() as tape:
        output = model(image, training=True)
        loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(output, labels, num_cls)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(total_loss)
    return loc_loss, conf_loss, mask_loss, seg_loss


def main(argv):
    # set fixed random seed, load config files
    tf.random.set_seed(RANDOM_SEED)

    # using mix precision or not
    if MIXPRECISION:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    # get params for model
    train_iter, input_size, num_cls, lrs_schedule_params, loss_params, parser_params, model_params = get_params(
        FLAGS.name)

    # -----------------------------------------------------------------
    # set up Grappler for graph optimization
    # Ref: https://www.tensorflow.org/guide/graph_optimization
    @contextlib.contextmanager
    def options(opts):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(opts)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    # -----------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Creating the model instance of YOLACT")
    model = Yolact(**model_params)

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.bias))

    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the dataloader from: %s..." % FLAGS.tfrecord_dir)
    dateset = ObjectDetectionDataset(dataset_name=FLAGS.name,
                                     tfrecord_dir=os.path.join(FLAGS.tfrecord_dir, FLAGS.name),
                                     anchor_instance=model.anchor_instance,
                                     **parser_params)
    train_dataset = dateset.get_dataloader(subset='train', batch_size=FLAGS.batch_size)
    valid_dataset = dateset.get_dataloader(subset='val', batch_size=1)
    # count number of valid data for progress bar
    # Todo any better way to do it?
    num_val = 0
    for _ in valid_dataset:
        num_val += 1
    # -----------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics, learning rate schedule
    lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(**lrs_schedule_params)
    logging.info("Initiate the Optimizer and Loss function...")
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=FLAGS.momentum)
    criterion = loss_yolact.YOLACTLoss(**loss_params)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
    seg = tf.keras.metrics.Mean('seg_loss', dtype=tf.float32)
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

    best_masks_map = 0.
    iterations = checkpoint.step.numpy()

    for image, labels in train_dataset:
        # check iteration and change the learning rate
        if iterations > train_iter:
            break

        checkpoint.step.assign_add(1)
        iterations += 1
        with options({'constant_folding': True,
                      'layout_optimize': True,
                      'loop_optimization': True,
                      'arithmetic_optimization': True,
                      'remapping': True}):
            loc_loss, conf_loss, mask_loss, seg_loss = train_step(model, criterion, train_loss, optimizer, image,
                                                                  labels, num_cls)
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
            tf.print("Iteration {}, LR: {}, Total Loss: {}, B: {},  C: {}, M: {}, S:{} ".format(
                iterations,
                optimizer._decayed_lr(var_dtype=tf.float32),
                train_loss.result(),
                loc.result(),
                conf.result(),
                mask.result(),
                seg.result()
            ))

        if iterations and iterations % FLAGS.save_interval == 0:
            # save checkpoint
            save_path = manager.save()
            logging.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            # validation and print mAP table
            all_map = evaluate(model, valid_dataset, num_val, num_cls)
            box_map, mask_map = all_map['box']['all'], all_map['mask']['all']
            tf.print(f"box mAP:{box_map}, mask mAP:{mask_map}")

            with test_summary_writer.as_default():
                tf.summary.scalar('Box mAP', box_map, step=iterations)
                tf.summary.scalar('Mask mAP', mask_map, step=iterations)

            # Saving the weights:
            if mask_map > best_masks_map:
                best_masks_map = mask_map
                model.save_weights(f'{FLAGS.weights}/weights_{FLAGS.name}_{str(best_masks_map)}.h5')

            # reset the metrics
            train_loss.reset_states()
            loc.reset_states()
            conf.reset_states()
            mask.reset_states()
            seg.reset_states()


if __name__ == '__main__':
    app.run(main)
