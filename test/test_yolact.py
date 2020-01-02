from models.yolact import Yolact

model = Yolact(fpn_channels=256, feature_map_size=[69, 35, 18, 9, 5], num_class=20, num_mask=4,
               aspect_ratio=[1, 0.5, 2], scale=1)
model.build(input_shape=(1, 550, 550, 3))
