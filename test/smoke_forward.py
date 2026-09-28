"""Quick smoke test: build YOLACT, run forward + detect on a dummy image."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from config import get_params
from yolact import Yolact


def main():
    _, _, _, _, _, _, model_params = get_params('coco')
    model = Yolact(**model_params)
    x = tf.random.uniform([1, 550, 550, 3])
    out = model(x, training=False)
    for k, v in out.items():
        print(f'{k}: {tuple(v.shape)}')
    dets = model.detect(out)
    print('detections batch size:', len(dets))
    print('smoke test OK')


if __name__ == '__main__':
    main()
