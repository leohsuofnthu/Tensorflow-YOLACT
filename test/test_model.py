import config as cfg
from yolact import Yolact

model = Yolact(**cfg.model_parmas, **cfg.anchor_params, **cfg.detection_params)
model.build(input_shape=(2, 550, 550, 3))

model.summary()
