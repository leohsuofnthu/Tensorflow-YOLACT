# Weights

Trained YOLACT checkpoints (`.h5`) are saved here by `train.py`.

This repository does **not** currently include pretrained COCO weights.
After training, load them like:

```bash
python eval.py --mode image --name coco --image ./demo.jpg --weights ./weights/weights_coco_XX.weights.h5 --output_dir ./results
```

You can also restore from `./checkpoints` by passing that directory to `--weights`.

Keras 3 requires weight files to end with `.weights.h5`.
