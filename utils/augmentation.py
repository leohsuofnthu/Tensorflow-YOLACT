import tensorflow as tf

from utils import utils
import config as cfg

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
Ref: https://github.com/dbolya/yolact/blob/821e83047847b9b1faf21b03b0d7ad521508f8ee/utils/augmentations.py
"""


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks=None, boxes=None, labels=None):
        for transform in self.transforms:
            image, masks, boxes, labels = transform(image, masks, boxes, labels)
        return image, masks, boxes, labels


class ConvertFromInts(object):
    """Convert Iamge to tf.float32 and normalize to [0, 1]"""

    def __init__(self):
        ...

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # convert to tf.float32, in range [0 ~ 1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, masks, boxes, labels


class RandomBrightness(object):
    # input image range: [0 ~ 1]
    def __init__(self, delta=0.125):
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=self.delta)
        return image, masks, boxes, labels


class RandomContrast(object):
    # input image range: [0 ~ 1]
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_contrast(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels


class RandomSaturation(object):
    # input image range: [0 ~ 1]
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_saturation(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels


class RandomHue(object):
    # input image range: [0 ~ 1]
    def __init__(self, delta=0.025):
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_hue(image, max_delta=self.delta)
        return image, masks, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.actions = [
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, masks, boxes, labels):
        image, masks, boxes, labels = self.rand_brightness(image, masks, boxes, labels)
        # different order have different effect
        if tf.random.uniform([1]) > 0.5:
            image, masks, boxes, labels = Compose(self.actions[:-1])(image, masks, boxes, labels)
        else:
            image, masks, boxes, labels = Compose(self.actions[1:])(image, masks, boxes, labels)
        return image, masks, boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        # exapnd the image with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            return image, masks, boxes, labels

        height, width, depth = image.get_shape()
        # expand 4 times at most
        ratio = tf.random.uniform([1], minval=1, maxval=4)
        tf.print(ratio)
        # define the leftmost, topmost coordinate for putting original image to expanding image
        left = tf.random.uniform([1], minval=0, maxval=(width * ratio - width))
        tf.print(left)
        top = tf.random.uniform([1], minval=0, maxval=(height * ratio - height))

        # padding the image, mask according to the left and top
        left_padding = int(left)
        top_padding = int(top)
        expand_width = int(width * ratio)
        expand_height = int(height * ratio)

        image = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(image, 0),
                                                        top_padding,
                                                        left_padding,
                                                        expand_height,
                                                        expand_width))
        masks = tf.squeeze(tf.image.pad_to_bounding_box(masks,
                                                        top_padding,
                                                        left_padding,
                                                        expand_height,
                                                        expand_width))
        # fill mean value of image
        offset = tf.constant(self.mean)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        mean_mask = tf.cast((image == 0), image.dtype) * offset
        image = image + mean_mask

        # recalculate the bbox [ymin, xmin, ymax, xmax]
        ymin = ((boxes[:, 0] * height) + top) / expand_height
        xmin = ((boxes[:, 1] * width) + left) / expand_width
        ymax = ((boxes[:, 2] * height) + top) / expand_height
        xmax = ((boxes[:, 3] * width) + left) / expand_width
        new_boxes = tf.stack([ymin, xmin, ymax, xmax])
        return image, masks, new_boxes, labels


class RandomSampleCrop(object):
    def __init__(self):
        self.min_iou = [0, 0.1, 0.3, 0.7, 0.9, 1]

    def __call__(self, image, masks, boxes=None, labels=None):
        # choose the min_object_covered value in self.sample_options
        idx = tf.cast(tf.random.uniform([1], minval=0, maxval=5.50), tf.int32)
        min_iou = self.min_iou[idx]
        tf.print(min_iou)
        if min_iou == 1:
            return image, masks, boxes, labels

        # Geometric Distortions (img, bbox, mask)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=min_iou,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.1, 1.0),
            max_attempts=50)

        # the distort box is the area of the cropped image, original image will be [0, 0, 1, 1]
        distort_bbox = distort_bbox[0, 0]

        # cropped the image
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])

        # cropped the mask
        bbox_begin = tf.concat([[0], bbox_begin], axis=0)
        bbox_size = tf.concat([[-1], bbox_size], axis=0)
        cropped_masks = tf.slice(masks, bbox_begin, bbox_size)
        cropped_masks.set_shape([None, None, None, 1])

        # resize the scale of bboxes for cropped image
        v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
        boxes = boxes - v
        s = tf.stack([distort_bbox[2] - distort_bbox[0],
                      distort_bbox[3] - distort_bbox[1],
                      distort_bbox[2] - distort_bbox[0],
                      distort_bbox[3] - distort_bbox[1]])
        boxes = boxes / s

        # filter out
        scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], boxes.dtype), boxes)
        bool_mask = scores > 0.5
        classes = tf.boolean_mask(labels, bool_mask)
        bboxes = tf.boolean_mask(boxes, bool_mask)

        # deal with negative value of bbox
        bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

        cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)

        return cropped_image, cropped_masks, bboxes, classes


class RandomMirror(object):
    def __int__(self):
        ...

    def __call__(self, image, masks, boxes, labels):
        # random mirroring with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)
            boxes = tf.stack([boxes[:, 0], 1 - boxes[:, 3],
                              boxes[:, 2], 1 - boxes[:, 1]], axis=-1)
        return image, masks, boxes, labels


class Resize(object):
    """Resize to certain size after augmentation"""

    def __init__(self, output_size, proto_output_size):
        self.output_size = output_size
        self.proto_output_size = proto_output_size

    def __call__(self, image, masks, boxes=None, labels=None):
        # resize the image to output size
        image = tf.image.resize(image, [self.output_size, self.output_size])

        # resize the mask to proto_out_size
        masks = tf.image.resize(tf.expand_dims(masks, -1), [self.proto_output_size, self.proto_output_size],
                                method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.cast(masks + 0.5, tf.int64)
        masks = tf.squeeze(masks)
        masks = tf.cast(masks, tf.float32)

        return image, masks, boxes, labels


class BackboneTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # normalize image (for Resnet), some model might only subtract mean, so modified for ur own need
        offset = tf.constant(self.mean)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset

        scale = tf.constant(self.std)
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, masks, boxes, labels


class SSDAugmentation(object):
    def __init__(self, mode, mean=cfg.MEANS, std=cfg.STD):
        if mode == 'train':
            self.augmentations = Compose([
                ConvertFromInts(),
                # PhotometricDistort(),
                # Expand(mean),
                # RandomSampleCrop(),
                # RandomMirror(),
                Resize(cfg.OUTPUT_SIZE, cfg.PROTO_OUTPUT_SIZE),
                # preserve aspect ratio or not?
                BackboneTransform(mean, std)
            ])
        else:
            # no data augmentation for validation and test set
            self.augmentations = Compose([
                ConvertFromInts(),
                Resize(cfg.OUTPUT_SIZE, cfg.PROTO_OUTPUT_SIZE),
                # preserve aspect ratio or not?
                BackboneTransform(mean, std)
            ])

    def __call__(self, image, masks, boxes, labels):
        return self.augmentations(image, masks, boxes, labels)
