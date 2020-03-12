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
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr

        self.learning_rate = tf.convert_to_tensor(warmup_lr)
        dtype = self.learning_rate.dtype
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.lr = tf.cast(initial_lr, dtype)
        self.decay_rate = 0.1

    def __call__(self, step):

        if step <= self.warmup_steps:
            # warm up stage
            self.learning_rate = (self.lr - self.warmup_lr) * (step / self.warmup_steps) + self.warmup_lr
        elif step == 280000:
            self.learning_rate *= self.decay_rate
        elif step == 600000:
            self.learning_rate *= self.decay_rate
        elif step == 700000:
            self.learning_rate *= self.decay_rate
        elif step == 750000:
            self.learning_rate *= self.decay_rate
        elif step > 750000:
            self.learning_rate *= self.decay_rate
        else:
            self.learning_rate *= 1
        return self.learning_rate

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_steps,
            "initial learning rate": self.initial_lr
        }

