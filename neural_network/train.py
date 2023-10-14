
import time


def train(network,
          inputs,
          targets,
          num_epochs,
          iterator,
          loss,
          optimizer,
          verbose=True):

    for epoch in range(num_epochs):
        # initial error loss for current epoch
        epoch_loss = 0.0
        start_time = time.time()
        for batch in iterator(inputs, targets):
            # perform forward step and get predicted probability
            predicted = network.forward(batch.inputs)

            # calculate loss over all elements in batch
            epoch_loss += loss.loss(predicted, batch.targets)

            # calculate gradients based predicted values
            grad = loss.grad(predicted, batch.targets)

            # run backprop
            network.backward(grad)

            # updating params in layers
            optimizer.step(network)

        # normalize loss
        epoch_loss /= len(inputs)

        if verbose:
            epoch_time = time.time() - start_time
            score = network.score(inputs, targets)
            print(f"epoch {epoch + 1} / {num_epochs} loss: {epoch_loss:.4f} score: {score:.4f} time: {epoch_time:.2f}")
