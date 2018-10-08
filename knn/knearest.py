import csv
from random import random
import math
from collections import defaultdict

dataset_filename = 'colors.csv'


def load_colors(filename):
    with open(filename) as dataset_file:
        lines = csv.reader(dataset_file)
        for line in lines:
            yield tuple(float(y) for y in line[0:3]), line[3]


def generate_colors(count=100):
    for i in range(count):
        yield (random(), random(), random())


def color_distance(color1, color2):
    channels = zip(color1, color2)
    sum_distance_squared = 0
    for c1, c2 in channels:
        sum_distance_squared += (c1 - c2) ** 2
    return math.sqrt(sum_distance_squared)


def nearest_neighbors(model_colors, num_neighbors):
    model = list(model_colors)
    target = yield
    while True:
        distances = sorted(
            ((color_distance(c[0], target), c) for c in model),
        )
        labels = defaultdict(int)
        for d in distances[0:num_neighbors]:
            print('distance : {:.2f} label {}'.format(d[0], d[1][1]))
            labels[d[1][1]] += 1

        max_label = sorted(labels, key=labels.get, reverse=True)[0]
        target = yield max_label


model_colors = load_colors(dataset_filename)
target_colors = generate_colors(3)
get_neighbors = nearest_neighbors(model_colors, 3)
next(get_neighbors)

for item, color in enumerate(target_colors):
    print('example {}'.format(item))
    print('color RGB: {:.2f} {:.2f} {:.2f}'.format(*color))
    label = get_neighbors.send(color)
    print('predicted label: {}'.format(label))
    print('')
