
"""
Differential evolution example
https://en.wikipedia.org/wiki/Differential_evolution
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def shifted_ewm(series, alpha, adjust=True):
    """ calculate exponential weighted mean of shifted data

        series - vector
        alpha - 0 < alpha < 1
    """
    return series.shift().ewm(alpha=alpha, adjust=adjust).mean()

def diff_evolution(series, adjust=False, eps=10e-5):
    """ find best signal using differential evolution algorithm """

    def func(alpha):
        """ calculate mean squared error between original and shifted series """
        # get shifted series
        shifted = shifted_ewm(series=series,
                              alpha=min(max(alpha, 0), 1),
                              adjust=adjust)
        # calculate mse
        mse = np.mean(np.power(series - shifted, 2))
        return mse

    # use scipy for find minimun of function
    optimized = differential_evolution(func=func,
                                       bounds=[(0 + eps, 1 - eps)])
    # check
    if not optimized['success']:
        # raise if optimization error
        raise Exception(optimized['message'])
    # return shifted series with optimized alpha value
    return shifted_ewm(series=series,
                       alpha=optimized['x'][0],
                       adjust=adjust).shift(-1)

if __name__ in '__main__':
    # set random seed
    np.random.seed(42)
    # set number of points
    points = 52
    # set amplitude of raw signal and noise signals
    ampl_raw, ampl_noise = 1, 2
    # set x var
    x = np.linspace(1, 52, points)
    # set y_true signal
    df = pd.Series(np.sin(x / 8) * ampl_raw, index=x, name='y_true').to_frame()
    # set noise signal
    df['y_noise'] = df['y_true'] + \
                        (np.random.rand(points) - 0.5) * ampl_noise + \
                        2 * ampl_noise * np.random.choice([-1, 0, 1],
                                                          size=points,
                                                          p=[0.02, 0.96, 0.02])
    # calculate differential evolution signal
    df['y_de'] = diff_evolution(df['y_noise'])
    # plot chart
    df.plot()
    plt.show()
