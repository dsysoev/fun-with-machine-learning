
import time
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
        start_time = time.time()
        for batch in iterator(inputs, targets):
            # perform forward step and get predicted probability
            predicted = network.predict_proba(batch.inputs)
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
            # updating params in layers
            optimizer.step(network)

        if verbose:
            epoch_time = time.time() - start_time

            network.training = False
            train_prediction = network.predict(inputs)
            network.training = True
            # calculate test accuracy
            train_targets = np.argmax(targets, axis=1)
            accuracy = np.mean(train_prediction == train_targets)
            print(f"epoch {epoch + 1} / {num_epochs} loss : {epoch_loss:.4f} accuracy {accuracy}")
