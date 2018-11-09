

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
        for batch in iterator(inputs, targets):
            # perform forward step and get predicted probability
            predicted = network.predict_proba(batch.inputs)
            # calculate loss over all elements in batch
            epoch_loss += loss.loss(predicted, batch.targets)
            # calculate gradients based predicted values
            grad = loss.grad(predicted, batch.targets)
            # run backprop
            network.backward(grad)
            # updating params in layers
            optimizer.step(network)
        if verbose:
            # print loss
            print("epoch: {:d} / {:d} loss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss))
