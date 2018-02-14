
"""
Example of using minimize function with cross validation method
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from smoothing import triple_exponential_smoothing

def cross_val_score(params, args):
    """ calculate mean squared error using
        triple exponential smoothing algorithm

        params - tuple with (alpha, beta, gamma) values
        args - dict with parameters
               {
                   'data': train test data,
                   'n_splits': number of splits,
                   'slen': length of season,
               }

    """
    values = args['data']
    alpha, beta, gamma = params
    # set number of splits
    tscv = TimeSeriesSplit(n_splits=args['n_splits'])
    # errors list for each split
    errors = []
    for train, test in tscv.split(values):
        # calculate train and predicted values
        vals = triple_exponential_smoothing(
            series=values[train],
            slen=args['slen'],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            n_preds=len(test))
        # get predicted values
        predictions = vals[-len(test):]
        # calculate error on test data
        error = mean_squared_error(values[test], predictions)
        errors.append(error)
    # return mean error for all splits
    return np.mean(errors)

if __name__ in '__main__':
    data = pd.read_csv('hour_online.csv', parse_dates=['Time'], index_col='Time')
    # split data to train and test
    test_size = 500
    train_X, test_X = data.Users.values[:-test_size], data.Users.values[-test_size:]
    # set params
    params = {
        'data': train_X,
        'slen': 24 * 7, # number of seasons 24 hour * 7 day of week
        'n_splits': 5,
    }
    # use minimize for find best alpha, beta and gamma values
    optimized = minimize(
        cross_val_score,
        args=params,
        x0=[0.0, 0.0, 0.0], # initial values alpha, beta, gamma
        method="TNC",
        bounds=((0, 1), (0, 1), (0, 1)),
        options={'maxiter': 100},)
    print(optimized)
    # calculate predictions with optimized parameters
    modelvals = triple_exponential_smoothing(
        series=train_X,
        slen=params['slen'],
        alpha=optimized.x[0],
        beta=optimized.x[1],
        gamma=optimized.x[2],
        n_preds=len(test_X),
        )
    # add data to DataFrame
    results = pd.DataFrame(data.Users.values, columns=['Users'])
    results['Train'] = modelvals[:-len(test_X)] + [float('NaN')] * len(test_X)
    results['Predicted'] = [float('NaN')] * len(train_X) + modelvals[-len(test_X):]
    results.plot()
    plt.show()
