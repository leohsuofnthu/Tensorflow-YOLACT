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

    def __call__(self, image, masks=None, boxes=None, labels=None):
        for transform in self.transforms:
            image, masks, boxes, labels = transform(image, masks, boxes, labels)
        return image, masks, boxes, labels


class ConvertFromInts(object):
    """Convert Image to tf.float32 and normalize to [0, 1]"""

    def __init__(self):
        ...

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # convert image from uint8 to float32
        preprocessed_img = tf.cast(image, tf.float32)
        return preprocessed_img, masks, boxes, labels


class BackbonePreprocess(object):
    """Preprocessed by corresponded backbone transformation"""

    def __init__(self, preprocess_func):
        self.preprocess_func = preprocess_func

    def __call__(self, image, masks=None, boxes=None, labels=None):
        preprocessed_img = self.preprocess_func(image)
        return preprocessed_img, masks, boxes, labels


class RandomBrightness(object):
    # input image range: [0.0 ~ 255.0]
    def __init__(self, delta=0.12):
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=self.delta)
        return image, masks, boxes, labels


class RandomContrast(object):
    # input image range: [0.0 ~ 255.0]
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
    # input image range: [0.0 ~ 255.0]
    def __init__(self, delta=0.08):
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
    """
    From https://github.com/FurkanOM/tf-ssd/blob/734bfd0cd1343b424bfad59c4b8c3cbef4775d86/utils/bbox_utils.py#L178
    """

    def __init__(self):
        ...

    def __call__(self, image, masks, boxes, labels):
        # exapnd the image with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            return image, masks, boxes, labels

        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)

        # expand 4 times at most
        expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
        final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
        pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
        pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
        pad_right = final_width - (width + pad_left)
        pad_bottom = final_height - (height + pad_top)

        mean, _ = tf.nn.moments(image, [0, 1])
        expanded_image = tf.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=-1)
        expanded_image = tf.where(expanded_image == -1, mean, expanded_image)
        expanded_masks = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(masks, -1),
                                                                 tf.cast(pad_top, tf.int32),
                                                                 tf.cast(pad_left, tf.int32),
                                                                 tf.cast(final_height, tf.int32),
                                                                 tf.cast(final_width, tf.int32)), -1)
        # recalculate the bbox [xmin, ymin, xmax, ymax]
        min_max = tf.stack([-pad_left, -pad_top, pad_right + width, pad_bottom + height], -1) / [width, height, width,
                                                                                                 height]
        x_min, y_min, x_max, y_max = tf.split(min_max, 4)
        renomalized_bboxes = boxes - tf.concat([x_min, y_min, x_min, y_min], -1)
        renomalized_bboxes /= tf.concat([x_max - x_min, y_max - y_min, x_max - x_min, y_max - y_min], -1)
        new_boxes = tf.clip_by_value(renomalized_bboxes, 0, 1)
        return expanded_image, expanded_masks, new_boxes, labels


# Todo I did a slightly different way for crop
class RandomSampleCrop(object):
    def __init__(self):
        self.min_iou = tf.constant([0.5, 0.6, 0.7, 0.8, 0.9, 1])

    def __call__(self, image, masks, boxes, labels):
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

        # deal with negative value of bbox
        bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

        # get new masks
        cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)

        return cropped_image, cropped_masks, bboxes, classes


class RandomMirror(object):
    # bbox [xmin, ymin, xmax, ymax]
    def __int__(self):
        ...

    def __call__(self, image, masks, boxes, labels=None):
        # random mirroring with probability 0.5
        if tf.random.uniform([1]) > 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(tf.expand_dims(masks, -1))
            boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1],
                              1 - boxes[:, 0], boxes[:, 3]], axis=-1)
            masks = tf.squeeze(masks, -1)
        return image, masks, boxes, labels


class Resize(object):
    """Resize to certain size after augmentation"""

    def __init__(self, output_size, proto_output_size, discard_w, discard_h):
        self.output_size = output_size
        self.proto_output_size = proto_output_size
        self.discard_w = discard_w
        self.discard_h = discard_h

    def __call__(self, image, masks, boxes, labels):
        # resize the image to output size
        image = tf.image.resize(image, [self.output_size, self.output_size],
                                method=tf.image.ResizeMethod.BILINEAR)

        # resize the mask to proto_out_size and binarize
        masks = tf.image.resize(tf.expand_dims(masks, -1), [self.proto_output_size, self.proto_output_size],
                                method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.cast(masks + 0.5, tf.int64)
        masks = tf.squeeze(masks)
        masks = tf.cast(masks, tf.float32)

        if tf.rank(masks) < 3:
            masks = tf.expand_dims(masks, axis=0)

        # discard the boxes that are too small
        w = self.output_size * (boxes[:, 2] - boxes[:, 0])  # xmax - xmin
        h = self.output_size * (boxes[:, 3] - boxes[:, 1])  # ymax - ymin

        # find intersection of those 2 idxs
        w_keep_idxs = tf.cast(w > self.discard_w, tf.int32)
        h_keep_idxs = tf.cast(h > self.discard_h, tf.int32)
        keep_idxs = w_keep_idxs * h_keep_idxs
        boxes = tf.boolean_mask(boxes, keep_idxs)
        masks = tf.boolean_mask(masks, keep_idxs)
        labels = tf.boolean_mask(labels, keep_idxs)

        return image, masks, boxes, labels


class SSDAugmentation(object):
    def __init__(self, mode, preprocess_func, mean, std, output_size, proto_output_size, discard_box_width,
                 discard_box_height):
        if mode == 'train':
            self.augmentations = Compose([
                ConvertFromInts(),
                # PhotometricDistort(),
                Expand(),
                # RandomSampleCrop(),
                # RandomMirror(),
                Resize(output_size, proto_output_size, discard_box_width, discard_box_height),
                # BackbonePreprocess(preprocess_func)
            ])
        else:
            # no data augmentation for validation and test set
            self.augmentations = Compose([
                ConvertFromInts(),
                # validation no need to resize mask tp proto size
                Resize(output_size, proto_output_size, discard_box_width, discard_box_height),
                BackbonePreprocess(preprocess_func),
            ])

    def __call__(self, image, masks, boxes, labels):
        return self.augmentations(image, masks, boxes, labels)
