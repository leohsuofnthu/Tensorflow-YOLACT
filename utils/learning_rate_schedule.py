import tensorflow as tf

"""
Custom learning rate scheduler for yolact 
"""


class Yolact_LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warmup_steps, warmup_lr, initial_lr):
        """
        :param warmup_steps:
        :param warmup_lr:
        :param initial_lr:
        """
        super(Yolact_LearningRateSchedule, self).__init__()
        self.warmup_step = warmup_steps
        self.warmup_lr = warmup_lr
        self.initial_lr = initial_lr

    def __call__(self, step):
        learning_rate = tf.convert_to_tensor(self.warmup_lr)
        dtype = learning_rate.dtype
        warmup_steps = tf.cast(self.warmup_step, dtype)
        lr = tf.cast(self.initial_lr, dtype)

        def f0():return (lr - self.warmup_lr) * (step / self.warmup_step) + self.warmup_lr
        def f1():return 1e-3
        def f2():return 1e-4
        def f3():return 1e-5
        def f4():return 1e-6
        def f5():return 1e-7

        learning_rate = tf.case([(tf.math.logical_and(tf.math.less(warmup_steps, step), tf.less_equal(step, 280000.)), f1),
                                 (tf.math.logical_and(tf.math.less(280000., step), tf.less_equal(step, 600000.)), f2),
                                 (tf.math.logical_and(tf.math.less(600000., step), tf.less_equal(step, 700000.)), f3),
                                 (tf.math.logical_and(tf.math.less(700000., step), tf.less_equal(step, 750000.)), f4),
                                 (tf.math.greater(step, 750000.), f5)],
                                default=f0,
                                exclusive=True)

        return learning_rate

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_steps,
            "initial learning rate": self.initial_lr
        }
