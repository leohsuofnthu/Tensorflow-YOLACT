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
            keys = tf.constant(keys)
            vals = tf.constant(vals)
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
        boxes = data['gt_bboxes']
        masks = data['gt_masks']
        is_crowds = data['gt_is_crowd']

        # if label_map is not none, remapping the class label, ex: COCO datasets
        if self.dict_tensor is not None:
            # todo clarify tf.int32 and tf.int64 on performance
            classes = tf.cast(classes, tf.int32)
            classes = self.dict_tensor.lookup(classes)
            classes = tf.cast(classes, tf.int64)

        # return original image for testing augmentation purpose
        original_img = tf.image.convert_image_dtype(tf.identity(image), tf.float32)
        original_img = tf.image.resize(original_img, [self.output_size, self.output_size])

        # put crowd annotation after non_crowd annotation
        crowd_idx = tf.where(is_crowds == True)[:, 0]
        non_crowd_idx = tf.where(tf.logical_not(is_crowds))[:, 0]
        idxs = tf.concat([non_crowd_idx, crowd_idx], axis=0)

        classes = tf.gather(classes, idxs)
        boxes = tf.gather(boxes, idxs)
        masks = tf.gather(masks, idxs)
        is_crowds = tf.gather(is_crowds, idxs)

        # Data Augmentation, Normalization, and Resize
        augmentor = SSDAugmentation(mode=mode, **self.augmentation_params)
        image, masks, norm_boxes, classes, is_crowds = augmentor(image, masks, boxes, classes, is_crowds)

        # Calculate num of crowd annotation here
        num_crowd = tf.reduce_sum(tf.cast(is_crowds, tf.int32))

        # remember to unnormalized the bbox
        boxes = norm_boxes * self.output_size

        # resized boxes for proto output size (for mask loss)
        boxes_norm = norm_boxes * self.proto_out_size

        # matching anchors
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            boxes, classes, num_crowd, **self.matching_params)

        if mode == 'train' and num_crowd > 0:
            # if num_crowd > 0:
            # tf.print(mode)
            # do not return annotation, cuz it s not used in loss calculation but evaluation
            masks = masks[:-num_crowd]
            classes = classes[:-num_crowd]
            boxes = boxes[:-num_crowd]
            boxes_norm = boxes_norm[:-num_crowd]

        # number of object in training sample
        num_obj = tf.shape(classes)[0]
        """
        if mode != 'train':
            tf.print("clssssssss", classes)
            tf.print("num obj", num_obj)
        """

        # Padding classes and mask to fix length [batch_size, num_max_fix_padding, ...]
        num_padding = self.num_max_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self.proto_out_size, self.proto_out_size])

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
            'num_crowd': num_crowd,
            'mask_target': masks,
            'max_id_for_anchors': max_id_for_anchors,
            'ori': original_img
        }
        return image, labels

    def _parse_train_data(self, data):
        return self._parse_common(data, 'train')

    def _parse_eval_data(self, data):
        return self._parse_common(data, 'val')

    def _parse_predict_data(self, data):
        return self._parse_common(data, 'test')
