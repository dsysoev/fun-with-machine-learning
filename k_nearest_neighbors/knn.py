
"""
Implementation of k-nearest neighbors algorithm
https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
"""

import csv
import math
import random
from collections import defaultdict


def load_colors(filename):
    """
    load colors dataset
    """
    with open(filename) as dataset_file:
        lines = csv.reader(dataset_file)
        for line in lines:
            yield tuple(float(y) for y in line[0:3]), line[3]


def generate_colors(count=100):
    """
    random color generator
    """
    for i in range(count):
        yield (random.random(), random.random(), random.random())


def color_distance(color1, color2):
    """
    calculate distance between two colors
    """
    channels = zip(color1, color2)
    sum_distance_squared = 0
    for c1, c2 in channels:
        sum_distance_squared += (c1 - c2) ** 2
    return math.sqrt(sum_distance_squared)


def k_nearest_neighbors(model_colors, num_neighbors):
    """
    k-nearest neighbors algorithm
    """
    model = list(model_colors)
    target = yield
    while True:
        distances = sorted(
            ((color_distance(c[0], target), c) for c in model),
        )
        labels = defaultdict(int)
        for d in distances[0:num_neighbors]:
            print('min distance: {:.2f} label: {}'.format(d[0], d[1][1]))
            labels[d[1][1]] += 1

        predicted_label = sorted(labels, key=labels.get, reverse=True)[0]
        target = yield predicted_label


if __name__ in '__main__':
    # load dataset
    model_colors = load_colors('colors.csv')
    # create instance of knn object
    get_predict = k_nearest_neighbors(model_colors, 3)
    next(get_predict)
    # create random colors
    target_colors = generate_colors(3)

    for color in target_colors:
        print('example color RGB: {:.2f} {:.2f} {:.2f}'.format(*color))
        label = get_predict.send(color)
        print('predicted label         : {}'.format(label))
        print('')
