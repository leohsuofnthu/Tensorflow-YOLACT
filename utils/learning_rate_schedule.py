import tensorflow as tf

"""
Custom learning rate scheduler for yolact 
"""


class Yolact_LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_lr, warmup_steps, warmup_lr, stages=None, stage_lrs=None):
        super(Yolact_LearningRateSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.initial_lr = initial_lr
        self.stages = [warmup_steps] + stages
        self.stages = self.stages[::-1]
        self.stage_lrs = stage_lrs[::-1]

    def __call__(self, step):
        def fn(s):
            return lambda: s

        learning_rate = tf.convert_to_tensor(self.warmup_lr)
        dtype = learning_rate.dtype
        lr = tf.cast(self.initial_lr, dtype)

        lrs_0 = lambda: (lr - self.warmup_lr) * (step / self.warmup_steps) + self.warmup_lr
        cases = [(tf.greater(step, x), fn(y)) for x, y in zip(self.stages, self.stage_lrs)]
        learning_rate = tf.case(cases, default=lrs_0, exclusive=False)

        return learning_rate

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_steps,
            "initial learning rate": self.initial_lr,
            "stages": self.stages,
            "lrs": self.stage_lrs
        }
