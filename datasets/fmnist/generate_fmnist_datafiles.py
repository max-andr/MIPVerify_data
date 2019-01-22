"""
Verified to work with Tensorflow 1.9

For Fashion MNIST, writes two separate `.mat` files containing the training and
test set respectively.

Pixel values are stored as uint8s (0-255); labels are zero-indexed.
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import scipy.io as sio

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

sio.savemat('fmnist_int_train.mat', {'images': x_train, 'labels': y_train})
sio.savemat('fmnist_int_test.mat', {'images': x_test, 'labels': y_test})