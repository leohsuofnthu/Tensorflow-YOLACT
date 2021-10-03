import tensorflow as tf


class ProtoNet(tf.keras.layers.Layer):
    """
        Creating the component of ProtoNet
        Arguments:
            num_prototype
    """

    def __init__(self, num_prototype):
        super(ProtoNet, self).__init__()
        self.upSampling = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.Conv1 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.Conv2 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.Conv3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.Conv4 = tf.keras.layers.Conv2D(256, 3, 1, "same")

        self.finalConv = tf.keras.layers.Conv2D(num_prototype, 1, 1, "valid")

    def call(self, p3):
        # (3,3) convolution * 3
        proto = tf.nn.relu(self.Conv1(p3))
        proto = tf.nn.relu(self.Conv2(proto))
        proto = tf.nn.relu(self.Conv3(proto))

        # upsampling + convolution
        proto = self.upSampling(proto)
        proto = tf.nn.relu(self.Conv4(proto))

        # final convolution
        proto = tf.nn.relu(self.finalConv(proto))
        return proto
