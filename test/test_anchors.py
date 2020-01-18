import tensorflow as tf
from data import anchor

d = tf.Variable([0.1, 0.6, 0.7, 0.3])


@tf.function
def test(x, pos, neg):
    if x < neg:
        return 0.
    elif x < pos:
        return -1.
    else:
        return 1.


p = tf.map_fn(lambda a: test(a, 0.5, 0.2), d)
tf.print(p)
