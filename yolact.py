"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
from layers.fpn import FeaturePyramidNeck
from layers.protonet import ProtoNet
from layers.head import PredictionModule
from utils.create_prior import make_priors
import tensorflow as tf

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self, input_size, fpn_channels, feature_map_size, num_class , num_mask, aspect_ratio, scales):
        super(Yolact, self).__init__()
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # use pre-trained ResNet50
        # Todo figure out how pre-trained can be train again
        base_model = tf.keras.applications.ResNet50(input_shape=(550, 550, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)
        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)
        self.predictionHead = []
        for idx, f_size in enumerate(feature_map_size):
            pred = PredictionModule(2, f_size, len(aspect_ratio), num_class, num_mask)
            self.predictionHead.append(pred)

    def call(self, inputs):
        # backbone(ResNet + FPN)
        c3, c4, c5 = self.backbone_resnet(inputs)
        print("c3: ", c3.shape)
        print("c4: ", c4.shape)
        print("c5: ", c5.shape)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        # Protonet branch
        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        print("protonet: ", protonet_out.shape)

        # Prediction Head branch
        prediction = []

        # Todo Share same prediction module and concate the output for each prediction to be [batch, num_anchor, ..]
        for idx, f_map in enumerate(fpn_out):
            preds = self.predictionHead[idx](f_map)
            print("p%s prediction:" % (idx+3))
            for i, p in enumerate(preds):
                print(p.shape)
            prediction.append(preds)
        print(len(prediction))

        # Todo concatenate each prediction (conf, loc, mask)
        return prediction, protonet_out
