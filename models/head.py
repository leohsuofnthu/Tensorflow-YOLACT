import tensorflow as tf

assert tf.__version__.startswith('2')


class PredictionModule(tf.keras.Model):
    """
        Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, out_channels, f_size, num_class, num_mask, aspect_ratio, scale):
        super(PredictionModule, self).__init__()
        self.anchors = self._make_priors(f_size, scale, aspect_ratio)
        self.num_anchors = len(self.anchors)
        self.num_class = num_class
        self.num_mask = num_mask
        self.initialConv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same", activation="relu")
        self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                                                activation="relu")
        self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same", activation="relu")
        self.maskConv = tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), 1, padding="same",
                                               activation="relu")

    def call(self, p):
        p = self.initialConv(p)
        p = self.initialConv(p)
        pred_class = self.classConv(p)
        pred_box = self.boxConv(p)
        pred_mask = self.maskConv(p)

        return [pred_class, pred_box, pred_mask]

    def _make_priors(self, size, scale, aspect_ratio):
        return [1]
        pass
