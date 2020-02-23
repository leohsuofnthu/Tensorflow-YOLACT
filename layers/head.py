import tensorflow as tf


class PredictionModule(tf.keras.layers.Layer):

    def __init__(self, out_channels, num_anchors, num_class, num_mask):
        super(PredictionModule, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class
        self.num_mask = num_mask

        self.Conv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same",
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")

        self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same",
                                              kernel_initializer=tf.keras.initializers.glorot_uniform())

        # activation of mask coef is tanh
        self.maskConv = tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), 1, padding="same",
                                               kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, p):
        p = self.Conv(p)

        pred_class = self.classConv(p)
        pred_box = self.boxConv(p)
        pred_mask = self.maskConv(p)

        # reshape the prediction head result for following loss calculation
        pred_class = tf.reshape(pred_class, [pred_class.shape[0], -1, self.num_class])
        pred_box = tf.reshape(pred_box, [pred_box.shape[0], -1, 4])
        pred_mask = tf.reshape(pred_mask, [pred_mask.shape[0], -1, self.num_mask])

        # add activation for conf and mask coef
        pred_mask = tf.keras.activations.tanh(pred_mask)

        return pred_class, pred_box, pred_mask
