"""
ref: https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/dataloader/tf_example_decoder.py#L63
"""
import tensorflow as tf


class TfExampleDecoder(object):
    def __init__(self):
        self._keys_to_features = {
            # I only take the key we need for this model
            'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/class/label_id': tf.io.VarLenFeature(dtype=tf.int64),
            'image/object/is_crowd': tf.io.VarLenFeature(dtype=tf.int64),
            'image/object/mask': tf.io.VarLenFeature(dtype=tf.string),
        }

    def _decode_image(self, parsed_tensors):
        image = tf.io.decode_jpeg(parsed_tensors['image/encoded'])
        image.set_shape([None, None, None])
        # convert gray_image to rgb
        image = tf.cond(tf.shape(image)[-1] < 3,
                        lambda: tf.image.grayscale_to_rgb(image),
                        lambda: tf.identity(image))
        return image

    def _decode_boxes(self, parsed_tensors):
        # pass the bbox in the order [xmin, ymin, xmax, ymax]
        xmin = parsed_tensors['image/object/bbox/xmin']
        ymin = parsed_tensors['image/object/bbox/ymin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32))

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features
        )

        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        masks = self._decode_masks(parsed_tensors)
        is_crowds = tf.cond(
            tf.greater(tf.shape(parsed_tensors['image/object/is_crowd']), 0),
            lambda: tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool),
            lambda: tf.zeros_like(parsed_tensors['image/object/class/label_id'], dtype=tf.bool))

        decoded_tensors = {
            'image': image,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'gt_classes': parsed_tensors['image/object/class/label_id'],
            'gt_is_crowd': is_crowds,
            'gt_bboxes': boxes,
            'gt_masks': masks,
            'gt_masks_png': parsed_tensors['image/object/mask']
        }

        return decoded_tensors
