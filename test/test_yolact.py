from yolact import Yolact

model = Yolact(input_size=550, fpn_channels=256, feature_map_size=[69, 35, 18, 9, 5], num_class=5, num_mask=4,
               aspect_ratio=[1, 0.5, 2], scales=[24, 48, 96, 192, 384])
model.build(input_shape=(1, 550, 550, 3))

model.summary()