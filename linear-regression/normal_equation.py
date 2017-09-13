
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def linear_regression_normal_equation(trainx, testx, trainy, testy, l2_value):
    """
    Linear Regression model with L2 regularization
    """

    def feed_dict(train, theta_value=None):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            return {X: trainxBias, y: trainy.reshape(-1, 1)}
        else:
            return {X: testxBias, y: testy.reshape(-1, 1), theta: theta_value}

    # add bias
    m, n = trainx.shape
    trainxBias = np.c_[np.ones((m, 1)), trainx]
    m, _ = testx.shape
    testxBias = np.c_[np.ones((m, 1)), testx]
    # create variable
    X = tf.placeholder(dtype=tf.float32, name="X")
    y = tf.placeholder(dtype=tf.float32, name="y")

    # regularization term
    lambda_value = tf.constant(l2_value, dtype=tf.float32)
    diagonal = tf.diag(tf.ones(n + 1, dtype=tf.float32))

    # calculate theta by normal equation
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(
        tf.matmul(XT, X) + lambda_value * diagonal), XT), y)

    # calculate predict value
    # add relu function for case when value below zero
    y_predict = tf.nn.relu(tf.matmul(X, theta))
    # calculate mean square log error
    msle = tf.reduce_mean(tf.square(tf.log1p(y_predict) - tf.log1p(y)))
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
