"""
Solve a classic problem (MNIST dataset) with neural networks.
https://en.wikipedia.org/wiki/MNIST_database
"""

import numpy as np

from train import train
from network import NeuralNetwork
from layers import Linear, Sigmoid, Relu, Tanh, LeakyRelu, Dropout
from optimizer import SGD
from data import BatchIterator, load_mnist, encode_labels


# set random seed
np.random.seed(137)
# load mnist data
train_inputs, train_targets = load_mnist('../data/', kind='train')
test_inputs, test_targets = load_mnist('../data/', kind='t10k')
# normalize input values using min max scaler style
train_inputs_normalized = train_inputs / 255.
test_inputs_normalzed = test_inputs / 255.
# one hot encoding labels
train_targets_enc = encode_labels(train_targets, 10)
# create a network architecture
# simple and fast
network = NeuralNetwork([
    Linear(input_size=784, output_size=32, batch_norm=False),
    Sigmoid(),
    Linear(input_size=32, output_size=32, batch_norm=False),
    Sigmoid(),
    Linear(input_size=32, output_size=784, batch_norm=False),
    Sigmoid(),
], training=True)
# train our network
train(
    network,
    train_inputs_normalized,
    train_inputs_normalized,
    num_epochs=10,
    iterator=BatchIterator(batch_size=4),
    optimizer=SGD(learning_rate=0.01),
    verbose=True
    )
# set prediction mode
network.training = False
# get test prediction
test_prediction = network.forward(test_inputs_normalzed)
np.savetxt('original.txt', test_inputs_normalzed, fmt='%f')
np.savetxt('prediction.txt', test_prediction, fmt='%f')
