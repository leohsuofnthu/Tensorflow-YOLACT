import tensorflow as tf

from utils import utils

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
Ref: https://github.com/dbolya/yolact/blob/821e83047847b9b1faf21b03b0d7ad521508f8ee/utils/augmentations.py
"""


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        for transform in self.transforms:
            image, masks, boxes, labels, is_crowds = transform(image, masks, boxes, labels, is_crowds)
        return image, masks, boxes, labels, is_crowds


class ConvertFromInts(object):
    """Convert Iamge to tf.float32 and normalize to [0, 1]"""

    def __init__(self):
        ...

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        # convert to tf.float32, in range [0 ~ 1]
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Todo Convert accroding to backbone
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, masks, boxes, labels, is_crowds


class RandomBrightness(object):
    # input image range: [0 ~ 1]
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=self.delta)
        return image, masks, boxes, labels, is_crowds


class RandomContrast(object):
    # input image range: [0 ~ 1]
    def __init__(self, lower=0.5, upper=0.6):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_contrast(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels, is_crowds


class RandomSaturation(object):
    # input image range: [0 ~ 1]
    def __init__(self, lower=0.5, upper=0.6):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_saturation(image, lower=self.lower, upper=self.upper)
        return image, masks, boxes, labels, is_crowds


class RandomHue(object):
    # input image range: [0 ~ 1]
    def __init__(self, delta=0.5):
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_hue(image, max_delta=self.delta)
        return image, masks, boxes, labels, is_crowds


class PhotometricDistort(object):
    def __init__(self):
        self.actions = [
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, masks, boxes, labels, is_crowds):
        image, masks, boxes, labels, is_crowds = self.rand_brightness(image, masks, boxes, labels, is_crowds)
        # different order have different effect
        if tf.random.uniform([1]) > 0.5:
            image, masks, boxes, labels, is_crowds = Compose(self.actions[:-1])(image, masks, boxes, labels, is_crowds)
        else:
            image, masks, boxes, labels, is_crowds = Compose(self.actions[1:])(image, masks, boxes, labels, is_crowds)
        return image, masks, boxes, labels, is_crowds


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels, is_crowds):
        # exapnd the image with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            return image, masks, boxes, labels, is_crowds

        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)

        # expand 4 times at most
        ratio = tf.squeeze(tf.random.uniform([1], minval=1, maxval=4))

        # define the leftmost, topmost coordinate for putting original image to expanding image
        left = tf.squeeze(tf.random.uniform([1], minval=0, maxval=(width * ratio - width)))
        top = tf.squeeze(tf.random.uniform([1], minval=0, maxval=(height * ratio - height)))

        # padding the image, mask according to the left and top
        left_padding = tf.cast(left, tf.int32)
        top_padding = tf.cast(top, tf.int32)
        expand_width = tf.cast(width * ratio, tf.int32)
        expand_height = tf.cast(height * ratio, tf.int32)

        image = tf.image.pad_to_bounding_box(image,
                                             top_padding,
                                             left_padding,
                                             expand_height,
                                             expand_width)

        masks = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(masks, -1),
                                                        top_padding,
                                                        left_padding,
                                                        expand_height,
                                                        expand_width), -1)

        # fill mean value of image
        offset = tf.constant(self.mean)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        mean_mask = tf.cast((image == 0), image.dtype) * offset
        image = image + mean_mask

        # recalculate the bbox [ymin, xmin, ymax, xmax]
        top = tf.cast(top, tf.float32)
        left = tf.cast(left, tf.float32)
        expand_height = tf.cast(expand_height, tf.float32)
        expand_width = tf.cast(expand_width, tf.float32)
        ymin = ((boxes[:, 0] * height) + top) / expand_height
        xmin = ((boxes[:, 1] * width) + left) / expand_width
        ymax = ((boxes[:, 2] * height) + top) / expand_height
        xmax = ((boxes[:, 3] * width) + left) / expand_width
        new_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        return image, masks, new_boxes, labels, is_crowds


# Todo I did a slightly different way for crop
class RandomSampleCrop(object):
    def __init__(self):
        self.min_iou = tf.constant([0.5, 0.6, 0.7, 0.8, 0.9, 1])

    def __call__(self, image, masks, boxes, labels, is_crowds):
        # choose the min_object_covered value in self.sample_options
        # idx = tf.cast(tf.random.uniform([1], minval=0, maxval=5.50), tf.int32)
        # min_iou = tf.squeeze(tf.gather(self.min_iou, idx))
        # if min_iou == 1:
        #     return image, masks, boxes, labels, is_crowds

        # Geometric Distortions (img, bbox, mask)
        boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=1)  # just in case
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=1,
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
        cropped_masks = tf.slice(tf.expand_dims(masks, -1), bbox_begin, bbox_size)
        cropped_masks.set_shape([None, None, None, 1])
        cropped_masks = tf.squeeze(cropped_masks, -1)

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
        is_crowds = tf.boolean_mask(is_crowds, bool_mask)

        # deal with negative value of bbox
        bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

        # get new masks
        cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)

        return cropped_image, cropped_masks, bboxes, classes, is_crowds


class RandomMirror(object):
    def __int__(self):
        ...

    def __call__(self, image, masks, boxes, labels=None, is_crowds=None):
        # random mirroring with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(tf.expand_dims(masks, -1))
            boxes = tf.stack([boxes[:, 0], 1 - boxes[:, 3],
                              boxes[:, 2], 1 - boxes[:, 1]], axis=-1)
            masks = tf.squeeze(masks, -1)
        return image, masks, boxes, labels, is_crowds


class Resize(object):
    """Resize to certain size after augmentation"""

    def __init__(self, output_size, proto_output_size, discard_w, discard_h):
        self.output_size = output_size
        self.proto_output_size = proto_output_size
        self.discard_w = discard_w
        self.discard_h = discard_h

    def __call__(self, image, masks, boxes, labels, is_crowds):
        # todo resize image mask while maintaining aspect ratio, also consider how to convert bbox

        # resize the image to output size
        image = tf.image.resize(image, [self.output_size, self.output_size],
                                method=tf.image.ResizeMethod.BILINEAR)

        # resize the mask to proto_out_size
        masks = tf.image.resize(tf.expand_dims(masks, -1), [self.proto_output_size, self.proto_output_size],
                                method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.cast(masks + 0.5, tf.int64)
        masks = tf.squeeze(masks)
        masks = tf.cast(masks, tf.float32)

        if tf.rank(masks) < 3:
            masks = tf.expand_dims(masks, axis=0)

        # discard the boxes that are too small
        w = self.output_size * (boxes[:, 3] - boxes[:, 1])  # xmax - xmin
        h = self.output_size * (boxes[:, 2] - boxes[:, 0])  # ymax - ymin

        # find intersection of those 2 idxs
        w_keep_idxs = tf.cast(w > self.discard_w, tf.int32)
        h_keep_idxs = tf.cast(h > self.discard_h, tf.int32)
        keep_idxs = w_keep_idxs * h_keep_idxs

        boxes = tf.boolean_mask(boxes, keep_idxs)
        masks = tf.boolean_mask(masks, keep_idxs)
        labels = tf.boolean_mask(labels, keep_idxs)
        is_crowds = tf.boolean_mask(is_crowds, keep_idxs)

        return image, masks, boxes, labels, is_crowds


class BackboneTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, masks=None, boxes=None, labels=None, is_crowds=None):
        # normalize image (for Resnet), some model might only subtract mean, so modified for ur own need
        # offset = tf.constant(self.mean)
        # offset = tf.expand_dims(offset, axis=0)
        # offset = tf.expand_dims(offset, axis=0)
        # image -= offset
        #
        # scale = tf.constant(self.std)
        # scale = tf.expand_dims(scale, axis=0)
        # scale = tf.expand_dims(scale, axis=0)
        # image /= scale
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, masks, boxes, labels, is_crowds


class SSDAugmentation(object):
    def __init__(self, mode, mean, std, output_size, proto_output_size, discard_box_width, discard_box_height):
        if mode == 'train':
            self.augmentations = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(mean),
                RandomSampleCrop(),
                RandomMirror(),
                Resize(output_size, proto_output_size, discard_box_width, discard_box_height),
                # BackboneTransform(mean, std)
            ])
        else:
            # no data augmentation for validation and test set
            self.augmentations = Compose([
                ConvertFromInts(),
                # validation no need to resize mask tp proto size
                Resize(output_size, proto_output_size, discard_box_width, discard_box_height),
                # preserve aspect ratio or not?
                # BackboneTransform(mean, std)
            ])

    def __call__(self, image, masks, boxes, labels, is_crowds):
        return self.augmentations(image, masks, boxes, labels, is_crowds)
