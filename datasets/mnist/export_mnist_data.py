"""
Verified to work with Tensorflow 1.9
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist 
import numpy as np
import scipy.io as sio

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# pad final dimension of dataset and normalize to between 0 and 1
x_train_f = np.expand_dims(x_train, 3)/255
x_test_f = np.expand_dims(x_test, 3)/255

sio.savemat('mnist_train.mat', {'images': x_train_f, 'labels': y_train})
sio.savemat('mnist_test.mat', {'images': x_test_f, 'labels': y_test})