
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import argparse

import numpy as np

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

    ax.set_title("""California housing with Linear Regression
    L2 regularization lambda = {}""".format(lambda_value))
    ax.set_xlabel('number of objects')
    ax.set_ylabel('MSLE')
    ax.set_xscale('log')
    ax.legend()
    plt.show()

def main():

    # create results file if it does not exist
    if FLAGS.force or not os.path.isfile(FLAGS.results_file):
        os.makedirs(os.path.dirname(FLAGS.results_file), exist_ok=True)

        # get data
        housing = fetch_california_housing()

        # create list of number of object
        num_objects_list = [50, 100, 500, 1000, 5000, 10000]
        lambda_value = FLAGS.lambda_value

        # collect data with different count of objects
        train_score_list, test_score_list, lambda_list = [], [], []
        for i in num_objects_list:
            # split data
            trainx, testx, trainy, testy = train_test_split(
                housing.data, housing.target, test_size=i, train_size=i,
                random_state=100)

            # get score
            train_score, test_score = linear_regression_normal_equation(
                trainx, testx, trainy, testy, lambda_value)

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
    main()
