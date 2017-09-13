
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
    results = pd.read_csv(FLAGS.results_file, index_col='lambda')
    # plot results
    ax = results.plot(alpha=0.2, legend=False)
    ax = results.rolling(10).mean().plot(ax=ax)
    ax.set_title('California housing Linear Regression with L2 regularization')
    ax.set_xlabel('lambda')
    ax.set_ylabel('MSLE')
    plt.show()

def main():

    # create results file if it does not exist
    if FLAGS.force or not os.path.isfile(FLAGS.results_file):
        os.makedirs(os.path.dirname(FLAGS.results_file), exist_ok=True)

        # create list of lambda of interest
        lambda_list = np.linspace(FLAGS.min_lambda,
                                  FLAGS.max_lambda,
                                  FLAGS.num_intervals)

        housing = fetch_california_housing()

        trainx, testx, trainy, testy = train_test_split(
            housing.data, housing.target, test_size=FLAGS.test_size)

        # collect data with different lambda value
        train_score_list, test_score_list = [], []
        for lambda_value in lambda_list:
            # get scores
            train_score, test_score = linear_regression_normal_equation(
                trainx, testx, trainy, testy, lambda_value)
            train_score_list.append(train_score[0])
            test_score_list.append(test_score[0])

        # create DataFrame object
        data = pd.DataFrame({'train_score': train_score_list,
                             'test_score': test_score_list},
                            index=lambda_list)
        # set num_objects as index
        data.index.name = 'lambda'
        # save data to csv file
        data.to_csv(FLAGS.results_file, header=True)

    plot_data()

if __name__ == '__main__':
    # eval filename without extention
    filename, _ = os.path.splitext(os.path.basename(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--min_lambda', type=float, default=0.)
    parser.add_argument('--max_lambda', type=float, default=2.)
    parser.add_argument('--num_intervals', type=int, default=200)
    parser.add_argument(
        '--results_file',
        type=str,
        default=os.path.join(tempfile.gettempdir(),
                             'fun-with-machine-learning',
                             filename + '.csv'), # output data has the same name
        help='File with results')

    FLAGS, unparsed = parser.parse_known_args()
    main()
