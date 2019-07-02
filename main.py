import numpy as np
import sys, os
import tensorflow as tf


if __name__ == '__main__':
    tf.nn.softmax
    with tf.compat.v1.Session() as sess:
        welcome = sess.run(tf.constant("Hello"))
        print(welcome)