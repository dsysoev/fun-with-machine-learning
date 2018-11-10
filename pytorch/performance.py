
"""
Comparison numpy vs pytorch
performance of random matrix multiplication  
"""

import sys
from time import time
from functools import wraps

import torch
import numpy as np


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        value = func(*args, **kwargs)
        end_ = int(round(time() * 1000)) - start
        return value, end_

    return _time_it


def numpy_mult(*shape):
    x = np.random.randn(*shape)
    y = np.random.randn(*shape)
    @measure
    def mult(x, y):
        return x @ y
    __, duration = mult(x, y)
    return duration


def torch_mult(*shape):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    @measure
    def mult(x, y):
        return x @ y
    __, duration = mult(x, y)
    return duration


if __name__ in '__main__':
    for func in [numpy_mult, torch_mult]:
        print('Multimply two random matrix using {}'.format(func.__name__))
        for shape in [1e2, 1e3, 2e3, 5e3]:
            args = (int(shape), int(shape))
            duration = func(*args)
            print('shape: {1:5d}x{1:<5d} wall time: {2} ms'.format(int(shape), duration))
