import tensorflow as tf


class ProtoNet(tf.keras.layers.Layer):
    """
        Creating the component of ProtoNet
        Arguments:
            num_prototype
    """

    def __init__(self, num_prototype):
        super(ProtoNet, self).__init__()
        self.Conv1 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation="relu")
        self.Conv2 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation="relu")
        self.Conv3 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation="relu")
        self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.Conv4 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation="relu")

        self.finalConv = tf.keras.layers.Conv2D(num_prototype, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation='relu')

    def call(self, p3):
        # (3,3) convolution * 3
        proto = self.Conv1(p3)
        proto = self.Conv2(proto)
        proto = self.Conv3(proto)

        # upsampling + convolution
        proto = self.upSampling(proto)
        proto = self.Conv4(proto)

        # final convolution
        proto = self.finalConv(proto)
        return proto
