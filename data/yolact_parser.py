from data import tfrecord_decoder
from utils import augmentation
import tensorflow as tf


class Parser(object):

    def __init__(self,
                 output_size,
                 anchor_instance,
                 match_threshold=0.5,
                 unmatched_threshold=0.5,
                 skip_crow_during_training=True,
                 use_bfloat16=True,
                 mode=None):

        self._mode = mode
        self._skip_crowd_during_training = skip_crow_during_training
        self._is_training = (mode == "train")

        self._example_decoder = tfrecord_decoder.TfExampleDecoder()

        self._output_size = output_size
        self._anchor_instance = anchor_instance
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold

        # Device.
        self._use_bfloat16 = use_bfloat16

        # Data is parsed depending on the model.
        if mode == "train":
            self._parse_fn = self._parse_train_data
        elif mode == "eval":
            self._parse_fn = self._parse_eval_data
        elif mode == "test":
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse_train_data(self, data):
        is_crowds = data['gt_is_crowd']
        classes = data['gt_classes']
        boxes = data['gt_bboxes']
        masks = data['gt_masks']

        # Skips annotations with `is_crowd` = True.
        # Todo: Need to understand control_dependeicies and tf.gather
        tf.print("Ignore crowd annotation")
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(input=classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(
                    pred=tf.greater(tf.size(input=is_crowds), 0),
                    true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
                    false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
            masks = tf.gather(masks, indices)

        image = data['image']

        print("classes", classes)

        # resize the image, box, mask
        tf.print("Resize image")
        image = tf.image.resize(image, [self._output_size, self._output_size])
        # Todo: resize boxes and masks

        # normalize the image
        tf.print("normalize image")
        image = tf.image.per_image_standardization(image)

        # data augmentation randomly
        print("data augmentation")
        image, boxes, masks = augmentation.random_augmentation(image, boxes, masks)

        # match anchors
        print("anchor matching")
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes, classes)

        # Todo check the shape of mask
        # label information need to be returned
        labels = {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'positiveness': match_positiveness,
            'mask_target': masks,
            # 'image_info': image_info
        }
        return image, labels

    def _parse_eval_data(self, data):
        pass

    def _parse_predict_data(self, data):
        pass
