
import numpy as np

from loss import MSE
from optimizer import SGD
from data import BatchIterator


def train(network,
          inputs,
          targets,
          num_epochs=5000,
          iterator=BatchIterator(),
          loss=MSE(),
          optimizer=SGD(),
          verbose=False):

    for epoch in range(num_epochs):
        # initial error loss for current epoch
        epoch_loss = 0.0
        grads_loss = {}
        for batch in iterator(inputs, targets):
            # perform forward step and get predicted probability
            predicted = network.predict_proba(batch.inputs)
            # print(network.get_layers_params())
            # calculate loss over all elements in batch
            epoch_loss += loss.loss(predicted, batch.targets)
            # calculate gradients based predicted values
            grad = loss.grad(predicted, batch.targets)
            # run backprop
            grads = network.backward(grad)
            # calculate mean gradient for each layer
            for num, grads_batch in grads.items():
                if num not in grads_loss:
                    grads_loss[num] = 0.
                grads_loss[num] += np.sum(np.abs(grads_batch))
            # print(np.sum(grad0))
            # updating params in layers
            optimizer.step(network)
        if verbose:
            print("epoch: {:d} / {:d}".format(epoch + 1, num_epochs))
            # print loss and mean gradients
            print("loss: {:10.4f}".format(epoch_loss), end=' ')
            print('mean grad: ', end='')
            for num in sorted(grads_loss.keys()):
                print("layer {} {:8.2f} ".format(num, grads_loss[num]), end='')
            print('')
