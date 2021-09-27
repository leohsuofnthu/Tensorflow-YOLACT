import tensorflow as tf

from data.coco_tfrecord_decoder import TfExampleDecoder
from utils.augmentation import SSDAugmentation


class Parser(object):
    def __init__(self, anchor_instance, mode=None, **parser_params):

        self._mode = mode
        self._is_training = (mode == "train")
        self._example_decoder = TfExampleDecoder()
        self._anchor_instance = anchor_instance
        self.output_size = parser_params['output_size']
        self.proto_out_size = parser_params['proto_out_size']
        self.num_max_padding = parser_params['num_max_padding']
        self.matching_params = parser_params['matching_params']
        self.augmentation_params = parser_params['augmentation_params']

        if parser_params['label_map'] is not None:
            keys = list(parser_params['label_map'].keys())
            vals = [parser_params['label_map'][k] for k in keys]
            keys = tf.constant(keys, dtype=tf.int64)
            vals = tf.constant(vals, dtype=tf.int64)
            self.dict_tensor = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys, vals), -1)
        else:
            self.dict_tensor = None

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

    def _parse_common(self, data, mode='train'):

        # The parse function parse single data only, not in batch (reminder for myself)
        image = data['image']
        classes = data['gt_classes']
        boxes = data['gt_bboxes']  # [xmin, ymin, xmax, ymax], normalized (0~1)
        masks = data['gt_masks']
        is_crowds = data['gt_is_crowd']
        num_obj = tf.shape(classes)[0]

        # return original image for testing augmentation purpose
        original_img = tf.image.convert_image_dtype(tf.identity(image), tf.float32)
        original_img = tf.image.resize(original_img, [self.output_size, self.output_size])

        # if label_map is not none, remapping the class label, ex: COCO datasets
        if self.dict_tensor is not None:
            classes = self.dict_tensor.lookup(classes)

        if mode == 'train':
            # ignore crowd annotation when training
            non_crowd_idx = tf.where(tf.logical_not(is_crowds))[:, 0]
            classes = tf.gather(classes, non_crowd_idx)
            boxes = tf.gather(boxes, non_crowd_idx)
            masks = tf.gather(masks, non_crowd_idx)

        # Data Augmentation, Normalization, and Resize
        augmentor = SSDAugmentation(mode=mode, **self.augmentation_params)
        image, masks, boxes, classes = augmentor(image, masks, boxes, classes)

        # matching anchors
        boxes = boxes * self.output_size
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            boxes, classes, **self.matching_params)
        max_gt_for_anchors = tf.gather(boxes * (self.proto_out_size / self.output_size), max_id_for_anchors)

        # Padding classes and mask to fix length [batch_size, num_max_fix_padding, ...]
        num_padding = self.num_max_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=classes.dtype)
        pad_boxes = tf.zeros([num_padding, 4], dtype=boxes.dtype)
        pad_masks = tf.zeros([num_padding, self.proto_out_size, self.proto_out_size], dtype=masks.dtype)

        masks = tf.concat([masks, pad_masks], axis=0)
        classes = tf.concat([classes, pad_classes], axis=0)
        boxes = tf.concat([boxes, pad_boxes], axis=0)

        labels = {
            'box_targets': box_targets,
            'cls_targets': cls_targets,
            'mask_target': masks,
            'bbox': boxes,
            'classes': classes,
            'positiveness': match_positiveness,
            'max_id_for_anchors': max_id_for_anchors,
            'max_gt_for_anchors': max_gt_for_anchors,
            'num_obj': num_obj,
            'ori': original_img
        }
        return image, labels

    def _parse_train_data(self, data):
        return self._parse_common(data, 'train')

    def _parse_eval_data(self, data):
        return self._parse_common(data, 'val')

    def _parse_predict_data(self, data):
        return self._parse_common(data, 'test')
