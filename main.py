import numpy as np
import sys, os
import tensorflow as tf


def main():
    with tf.Session() as sess:
        welcome = sess.run(tf.constant("Hello"))
        print(welcome)


if __name__ == '__main__':
    with tf.Session() as sess:
        welcome = sess.run(tf.constant("Hello"))
        print(welcome)