
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import argparse

import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

def linear_regression_normal_equation(dataX, dataY, l2, test_size):
    """
    Linear Regression model with L2 regularization
    """

    def feed_dict(train, theta_value=None):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = train_X_with_bias, trainY.reshape(-1, 1)
            return {X: xs, y: ys}
        else:
            xs, ys = test_X_with_bias, testY.reshape(-1, 1)
            return {X: xs, y: ys, theta: theta_value}

    # split data on train and test part
    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, test_size=test_size)

    # add bias
    m, n = trainX.shape
    train_X_with_bias = np.c_[np.ones((m, 1)), trainX]
    m, _ = testX.shape
    test_X_with_bias = np.c_[np.ones((m, 1)), testX]

    # create variable
    X = tf.placeholder(dtype=tf.float32, name="X")
    y = tf.placeholder(dtype=tf.float32, name="y")

    # regularization term
    lambda_value = tf.constant(l2, dtype=tf.float32)
    diagonal = tf.diag(tf.ones(n + 1, dtype=tf.float32))

    # calculate theta by normal equation
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(
        tf.matmul(XT, X) + lambda_value * diagonal), XT), y)

    # calculate predict value
    # add relu function for case when value below zero
    y_ = tf.nn.relu(tf.matmul(X, theta))
    # calculate mean square log error
    msle = tf.reduce_mean(tf.square(tf.log1p(y_) - tf.log1p(y)))

    # set empty data
    train_score_list, test_score_list = [], []

    with tf.Session() as sess:
        # calculate score on train data
        msle_train, theta_train = sess.run([msle, theta], feed_dict(True))
        train_score_list.append(msle_train)

        # calculate score on test data
        # use theta_train
        msle_test = sess.run(msle, feed_dict(False, theta_train))
        test_score_list.append(msle_test)

    return train_score_list, test_score_list
