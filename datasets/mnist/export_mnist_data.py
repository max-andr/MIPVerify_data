from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import scipy.io as sio
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
Exports the MNIST test set, with the images resized from 784 to 28*28*1
"""
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)

# None indicates that the first dimension, corresponding to the batch size, can be of any size.
x = tf.placeholder(tf.float32, [None, 784], name='original_image')

# With tf.reshape, size of dimension with special value -1 computed so total size remains constant.
x_image = tf.reshape(x, [-1,28,28,1], name='flattened_image')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sio.savemat('mnist_test.mat', {
        'images': sess.run(x_image, feed_dict={x: mnist.test.images}), 
        'labels': np.ndarray.tolist(mnist.test.labels)
        })
        
    sio.savemat('mnist_train.mat', {
        'images': sess.run(x_image, feed_dict={x: mnist.train.images}), 
        'labels': np.ndarray.tolist(mnist.train.labels)
        })

tf.app.run()