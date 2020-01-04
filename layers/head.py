import tensorflow as tf


class PredictionModule(tf.keras.layers.Layer):
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

    def __init__(self, out_channels, f_size, num_anchors, num_class, num_mask):
        super(PredictionModule, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class
        self.num_mask = num_mask

        self.Conv1 = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same", activation="relu")
        self.Conv2 = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same", activation="relu")

        self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                                                activation="relu")
        self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same", activation="relu")
        self.maskConv = tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), 1, padding="same",
                                               activation="relu")

    def call(self, p):
        p = self.Conv1(p)
        p = self.Conv2(p)

        pred_class = self.classConv(p)
        pred_box = self.boxConv(p)
        pred_mask = self.maskConv(p)

        # reshape the prediction head result for following loss calculation
        pred_class = tf.reshape(pred_class, [pred_class.shape[0], -1, self.num_class])
        pred_box = tf.reshape(pred_box, [pred_box.shape[0], -1, 4])
        pred_mask = tf.reshape(pred_mask, [pred_mask.shape[0], -1, self.num_mask])

        return [pred_class, pred_box, pred_mask]
