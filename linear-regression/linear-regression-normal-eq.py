
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

def get_score(dataX, dataY):
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

    # split data on te=rain and test part
    trainX, testX, trainY, testY = train_test_split(dataX, dataY,
                                                    test_size=FLAGS.test_size)

    # add bias
    m, n = trainX.shape
    train_X_with_bias = np.c_[np.ones((m, 1)), trainX]
    m, _ = testX.shape
    test_X_with_bias = np.c_[np.ones((m, 1)), testX]

    # create variable
    X = tf.placeholder(dtype=tf.float32, name="X")
    y = tf.placeholder(dtype=tf.float32, name="y")

    # create list of lambda of interest
    lambda_list = np.linspace(FLAGS.min_lambda, FLAGS.max_lambda,
                              FLAGS.num_intervals)

    # regularization term
    lambda_value = tf.Variable(0, dtype=tf.float32)
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
    results_data = {'lambda': [], 'train': [], 'test': []}

    with tf.Session() as sess:
        for lambda_val in lambda_list:

            # assign lambda
            val = sess.run(lambda_value.assign(lambda_val))
            results_data['lambda'].append(val)

            # calculate score on train data
            msle_train, theta_train = sess.run([msle, theta], feed_dict(True))
            results_data['train'].append(msle_train)

            # calculate score on test data
            # use theta_train
            msle_test = sess.run(msle, feed_dict(False, theta_train))
            results_data['test'].append(msle_test)

    # put data to DataFrame
    return pd.DataFrame(results_data).set_index('lambda')

def plot_data():
    """ plot chart from prepared data """

    # check results file
    if not os.path.isfile(FLAGS.results_file):
        raise IOError("No such file '{}'".format(FLAGS.results_file))

    # read DataFrame from results file
    results = pd.read_csv(FLAGS.results_file, index_col='lambda')
    # plot results
    ax = results.plot(alpha=0.5)
    ax.set_title('California housing Linear Regression with L2 regularization')
    ax.set_xlabel('lambda')
    ax.set_ylabel('MSLE')
    ax.legend()
    plt.show()

def main(_):

    # create results file if it does not exist
    if FLAGS.force or not os.path.isfile(FLAGS.results_file):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.results_file))
        housing = fetch_california_housing()
        data = get_score(housing.data, housing.target)
        data.to_csv(FLAGS.results_file, header=True)

    plot_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--min_lambda', type=float, default=0.)
    parser.add_argument('--max_lambda', type=float, default=2.)
    parser.add_argument('--num_intervals', type=int, default=100)
    parser.add_argument(
        '--results_file',
        type=str,
        default=os.path.join(tempfile.gettempdir(),
                             'fun-with-tf',
                             'linear-regression-normal-eq.csv'),
        help='File with results')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
