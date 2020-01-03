from utils.create_prior import make_priors

img_size = 550
f_size = [69, 35, 18, 9, 5]
aspect_ratio = [1, 0.5, 2]
scales = [24, 48, 96, 192, 384]

_, box = make_priors(img_size, f_size, aspect_ratio, scales)
print(box)
