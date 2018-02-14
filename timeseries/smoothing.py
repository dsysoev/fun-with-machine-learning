
"""
Implementation of several averaging algorithms:

- [Arithmetic mean](https://en.wikipedia.org/wiki/Arithmetic_mean)
- [Moving average](https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average)
- [Weighted moving average](https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average)
- [Basic exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing#Basic_exponential_smoothing)
- [Double exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing)
- [Triple exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing#Triple_exponential_smoothing)

"""

from __future__ import print_function

import pandas as pd
from matplotlib import pyplot as plt


def simple_average(series):
    """ calculate list with mean values """
    total = 0
    for value in series:
        total += value
    total /= len(series)
    return [total] * len(series)

def moving_average(series, window):
    """ calculate list with moving avarage values

        window - window averaging size
    """
    total = sum(series[:window])
    # set NaN for the first undefined values
    lst = [float('NaN')] * (window - 1) + [total / window]
    for pos in range(window, len(series)):
        total += series[pos] - series[pos - window]
        lst.append(total / window)
    return lst

def weighted_average(series, weights):
    """ calculate list with weighted moving average values

        weights - list of weigths for averaging
    """
    weights_sum = sum(weights)
    # normalize the weights and reverse it
    weigths_norm = [weight / weights_sum for weight in reversed(weights)]
    num_weights = len(weights)
    # set NaN for the first undefined values
    lst = [float('NaN')] * (num_weights - 1)
    for pos in range(num_weights, len(series) + 1):
        # calculate value for current time frame
        value = sum([v * w for v, w in zip(series[pos-num_weights:pos], weigths_norm)])
        lst.append(value)
    return lst

def exponential_smoothing(series, alpha):
    """ calculate list with exponential smoothing values

        alpha - level coeff, define mean value of series
    """
    # zero element in list given from series
    lst = [series[0]]
    for pos in range(1, len(series)):
        # y^{hat}_{n} = alpha * y_{n} + (1 - alpha) * y^{hat}_{n-1}
        lst.append(alpha * series[pos] + (1 - alpha) * lst[pos - 1])
    return lst

def double_exponential_smoothing(series, alpha, beta):
    """ calculate list with double exponential smoothing values

        alpha - level coeff, define mean value of series
        beta - trend coeff, define first derivative of series
    """
    # zero element in list given from series
    lst = [series[0]]
    len_series = len(series)
    for pos in range(1, len_series):
        if pos == 1:
            # calculate initial level and trend values
            level, trend = series[0], series[1] - series[0]
        if pos >= len_series:
            # predict value
            value = lst[-1]
        else:
            # get current value
            value = series[pos]
        # calculate new level based on exponential smoothing algorithm and
        # set new level to level and last level to last_level
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        # the same for trend
        trend = beta * (level - last_level) + (1 - beta) * trend
        lst.append(level + trend)
    return lst

def initial_trend(series, seasonlen):
    """ calculate initial trent for given series """
    if seasonlen * 2 > len(series):
        raise ValueError('length of series ({}) \
    shorter than double season length {}'.format(len(series), 2 * seasonlen))
    total = 0.
    for i in range(seasonlen):
        # add seasonal trend (first derivative)
        total += (series[i + seasonlen] - series[i]) / seasonlen
    # calculate mean trend for all seasons
    return total / seasonlen

def initial_seasonal_components(series, slen):
    """ calculate initial seasonal component for given series """
    seasonals = {}
    season_averages = []
    # calculate number of seasons
    n_seasons = len(series) // slen
    if not n_seasons:
        raise ValueError('season length {} \
    more than elements in series {}'.format(slen, len(series)))
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen * j:slen * j + slen]) / slen)
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen * j + i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    """ implementation of Holt-Winters algorithm

        calculate new list using Holt-Winters algorithm
        with given alpha, beta and gamma values
        add forecasting values to list if n_preds more than zero

        slen - length of season
        alpha - level coeff, define mean value of series
        beta - trend coeff, define first derivative of series
        gamma - season coeff, define seasonal of series
        n_preds - number of points for forecasting

    """
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series) + n_preds):
        if i == 0:
            # initial values
            level = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        num_season = i % slen
        if i >= len(series):
            # we are forecasting
            m = i - len(series) + 1
            result.append((level + m * trend) + seasonals[num_season])
        else:
            val = series[i]
            # calculate new level based on exponential smoothing algorithm and
            # set new level to level and last level to last_level
            last_level, level = level, alpha * (val - seasonals[num_season]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonals[num_season] = gamma * (val - level) + (1 - gamma) * seasonals[num_season]
            result.append(level + trend + seasonals[num_season])
    return result


if __name__ in '__main__':
    # read data
    data = pd.read_csv('hour_online.csv', parse_dates=['Time'], index_col='Time')
    data['simple_average'] = data.Users.agg(simple_average)
    data['moving_average'] = data.Users.agg(moving_average, window=24)
    # first value in weights associate
    # with last value of series
    weigths = [0.3, 0.2, 0.2, 0.2, 0.1]
    data['weighted_average'] = data.Users.agg(
        weighted_average, weights=weigths)
    data['exponential_smoothing'] = data.Users.agg(
        exponential_smoothing, alpha=0.95)
    data['double_exponential_smoothing'] = data.Users.agg(
        double_exponential_smoothing, alpha=0.95, beta=0.5)
    data['triple_exponential_smoothing'] = data.Users.agg(
        triple_exponential_smoothing,
        slen=24 * 7,
        alpha=0.06,
        beta=0.0,
        gamma=0.04,
        n_preds=0)
    # plot chart
    data.plot()
    plt.show()
