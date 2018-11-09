"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
https://en.wikipedia.org/wiki/Exclusive_or
"""

import numpy as np

from train import train
from network import NeuralNetwork
from layers import Linear, Relu
from optimizer import SGD


inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])
# set random seed
np.random.seed(37)
# create a network architecture
network = NeuralNetwork([
    Linear(input_size=2, output_size=4),
    Relu(),
    Linear(input_size=4, output_size=2)
])
# train our network
train(network,
      inputs,
      targets,
      num_epochs=300,
      optimizer=SGD(learning_rate=0.01),
      verbose=True)
# get prediction
prediction = network.predict(inputs)
print('prediction: {}'.format(prediction))
print('target    : {}'.format(np.argmax(targets, axis=1)))
