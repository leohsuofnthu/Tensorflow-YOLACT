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
        decay_rate = 0.1

        if step < warmup_steps:
            # warm up stage
            learning_rate = (lr - self.warmup_lr) * (step / self.warmup_step) + self.warmup_lr
        elif step == warmup_steps:
            learning_rate = lr
        elif step == 280000:
            learning_rate *= decay_rate
        elif step == 600000:
            learning_rate *= decay_rate
        elif step == 700000:
            learning_rate *= decay_rate
        elif step == 750000:
            learning_rate *= decay_rate
        elif step > 750000:
            learning_rate *= decay_rate
        else:
            learning_rate *= 1
        return learning_rate

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_step,
            "initial learning rate": self.initial_lr
        }
