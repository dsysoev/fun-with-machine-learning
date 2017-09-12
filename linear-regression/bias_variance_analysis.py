
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

from normal_equation import linear_regression_normal_equation

import matplotlib
matplotlib.style.use('seaborn')

def plot_data():
    """ plot chart from prepared data """

    # check results file
    if not os.path.isfile(FLAGS.results_file):
        raise IOError("No such file '{}'".format(FLAGS.results_file))

    # read DataFrame from results file
    results = pd.read_csv(FLAGS.results_file, index_col='num_objects')
    lambda_value = results['lambda'].unique()[0]
    results = results.drop('lambda', axis=1)
    # plot results
    ax = results.plot(alpha=1)

    ax.set_title("""California housing
        Linear Regression with L2 regularization lambda = {}""".format(
        lambda_value))
    ax.set_xlabel('number of objects')
    ax.set_ylabel('MSLE')
    ax.set_xscale('log')
    ax.legend()
    plt.show()

def main(_):

    # create results file if it does not exist
    if FLAGS.force or not os.path.isfile(FLAGS.results_file):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.results_file))

        # get data
        housing = fetch_california_housing()

        # create list of number of object
        num_objects_list = [50, 100, 500, 1000, 5000, 10000, 20000]
        lambda_value = FLAGS.lambda_value

        # add shuffle for data
        index_list = np.arange(housing.data.shape[0])
        np.random.seed(100)
        np.random.shuffle(index_list)

        # collect data with different count of objects
        train_score_list, test_score_list, lambda_list = [], [], []
        for i in num_objects_list:
            train_score, test_score = linear_regression_normal_equation(
                housing.data[index_list[:i]],
                housing.target[index_list[:i]],
                lambda_value,
                FLAGS.test_size)
            train_score_list.append(train_score[0])
            test_score_list.append(test_score[0])
            lambda_list.append(lambda_value)

        # create DataFrame object
        data = pd.DataFrame({'lambda': lambda_list,
                             'train_score': train_score_list,
                             'test_score': test_score_list},
                             index=num_objects_list)
        # set num_objects as index
        data.index.name = 'num_objects'
        # save data to csv file
        data.to_csv(FLAGS.results_file, header=True)

    plot_data()

if __name__ == '__main__':

    # eval filename without extention
    filename, _ = os.path.splitext(os.path.basename(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--lambda_value', type=float, default=0.35)
    parser.add_argument(
        '--results_file',
        type=str,
        default=os.path.join(tempfile.gettempdir(),
                             'fun-with-machine-learning',
                             filename + '.csv'), # output data has the same name
        help='File with results')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
