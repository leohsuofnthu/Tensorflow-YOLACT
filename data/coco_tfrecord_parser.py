import tensorflow as tf

from data.coco_tfrecord_decoder import TfExampleDecoder
from utils.augmentation import SSDAugmentation
from utils.utils import normalize_image


class Parser(object):
    def __init__(self,
                 output_size,
                 proto_output_size,
                 anchor_instance,
                 match_threshold=0.5,
                 unmatched_threshold=0.4,
                 num_max_fix_padding=100,
                 skip_crow_during_training=True,
                 use_bfloat16=True,
                 mode=None):

        self._mode = mode
        self._skip_crowd_during_training = skip_crow_during_training
        self._is_training = (mode == "train")

        self._example_decoder = TfExampleDecoder()

        self._output_size = output_size
        self._anchor_instance = anchor_instance
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold

        # output related
        # for classes and mask to be padded to fix length
        self._num_max_fix_padding = num_max_fix_padding
        # resize the mask to proto output size in advance (always 138, from paper's figure)
        self._proto_output_size = proto_output_size

        # Device (mix precision?)
        self._use_bfloat16 = use_bfloat16

        # Data is parsed depending on the model.
        if mode == "train":
            self._parse_fn = self._parse_train_data
        elif mode == "val":
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
        # The parse function parse single data only, not in batch (reminder for myself)
        image = data['image']
        classes = data['gt_classes']
        boxes = data['gt_bboxes']
        masks = data['gt_masks']
        is_crowds = data['gt_is_crowd']

        # Skips annotations with `is_crowd` = True.
        if self._skip_crowd_during_training and self._is_training:
            num_groundtruths = tf.shape(classes)[0]
            with tf.control_dependencies([num_groundtruths, is_crowds]):
                indices = tf.cond(
                    pred=tf.greater(tf.size(is_crowds), 0),
                    true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
                    false_fn=lambda: tf.cast(tf.range(num_groundtruths), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
            masks = tf.gather(masks, indices)

        # read and normalize the image, for testing augmentation
        original_img = tf.identity(image)

        # Data Augmentation, Normalization, and Resize
        # Todo output_size and masks proto_size, by getting from config file
        # Todo Make SSDAugmentation as singleton somewhere
        augmentor = SSDAugmentation(mode='train')
        image, masks, boxes, classes = augmentor(image, masks, boxes, classes)

        """

        # resize mask
        masks = tf.expand_dims(masks, axis=-1)
        masks = tf.image.resize(masks, [self._output_size, self._output_size], method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.cast(masks + 0.5, tf.int64)
        # masks = tf.squeeze(masks)
        # masks = tf.cast(masks, tf.float32)


        # remember to unnormalized the bbox
        boxes = boxes * self._output_size

        # resized boxes for proto output size
        boxes_norm = boxes * (self._proto_output_size / self._output_size)
        """

        # number of object in training sample
        num_obj = tf.size(classes)

        # matching anchors
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes, classes)

        # Padding classes and mask to fix length [batch_size, num_max_fix_padding, ...]
        num_padding = self._num_max_fix_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self._proto_output_size, self._proto_output_size])

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, pad_masks], axis=0)
        classes = tf.concat([classes, pad_classes], axis=0)
        boxes = tf.concat([boxes, pad_boxes], axis=0)
        boxes_norm = tf.concat([boxes_norm, pad_boxes], axis=0)

        labels = {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'bbox': boxes,
            'bbox_for_norm': boxes_norm,
            'positiveness': match_positiveness,
            'classes': classes,
            'num_obj': num_obj,
            'mask_target': masks,
            'max_id_for_anchors': max_id_for_anchors,
            'ori': original_img
        }
        return image, labels

    def _parse_eval_data(self, data):
        image = data['image']
        classes = data['gt_classes']
        boxes = data['gt_bboxes']
        masks = data['gt_masks']
        is_crowds = data['gt_is_crowd']

        # Skips annotations with `is_crowd` = True.
        # Todo: Need to understand control_dependeicies and tf.gather
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

        # convert image to range [0, 1], faciliate augmentation
        image = normalize_image(image)

        # we already resize the image when creating tfrecord
        image = tf.image.resize(image, [self._output_size, self._output_size])

        # resize mask
        masks = tf.expand_dims(masks, axis=-1)
        # using nearest neighbor to make sure the mask still in binary
        masks = tf.image.resize(masks, [self._proto_output_size, self._proto_output_size],
                                method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.cast(masks + 0.5, tf.int64)
        masks = tf.squeeze(tf.cast(masks, tf.float32))

        # resize boxes for resized image
        boxes = boxes * self._output_size

        # number of object in training sample
        num_obj = tf.size(classes)

        # resized boxes for proto output size
        boxes_norm = boxes * (self._proto_output_size / self._output_size)

        # matching anchors
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes, classes)

        # Padding classes and mask to fix length [None, num_max_fix_padding, ...]
        num_padding = self._num_max_fix_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self._proto_output_size, self._proto_output_size])

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, pad_masks], axis=0)
        classes = tf.concat([classes, pad_classes], axis=0)
        boxes = tf.concat([boxes, pad_boxes], axis=0)
        boxes_norm = tf.concat([boxes_norm, pad_boxes], axis=0)

        labels = {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'bbox': boxes,
            'bbox_for_norm': boxes_norm,
            'positiveness': match_positiveness,
            'classes': classes,
            'num_obj': num_obj,
            'mask_target': masks,
            'max_id_for_anchors': max_id_for_anchors
        }
        return image, labels

    def _parse_predict_data(self, data):
        pass
