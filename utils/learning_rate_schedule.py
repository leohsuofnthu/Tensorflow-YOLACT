import tensorflow as tf

"""
Custom learning rate scheduler for yolact 
"""


class Yolact_LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warmup_steps, warmup_lr, event_steps, event_lr):
        """
        :param warmup:
        :param warmup_lr:
        :param event_steps:
        :param event_lr:
        """
        super(Yolact_LearningRateSchedule, self).__init__()
        self.warmup_step = warmup_steps
        self.warmup_lr = warmup_lr
        self.event_steps = event_steps
        self.event_lr = event_lr

    def __call__(self, step):
        initial_learning_rate = tf.convert_to_tensor(self.warmup_lr)
        dtype = initial_learning_rate.dtype
        warmup_steps = tf.cast(self.warmup_step, dtype)
        event_steps = tf.cast(self.event_steps, dtype)
        event_lr = tf.cast(self.event_lr, dtype)
        # TODO finish the learning rate schedule
        return initial_learning_rate

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_step,
            "event steps": self.event_steps,
            "event learning rate": self.event_lr
        }
