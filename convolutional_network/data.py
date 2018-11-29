
import numpy as np
from six.moves import cPickle as pickle


def load_batch(filename='data_batch_1'):
    filepath = 'data-cifar10/cifar-10-batches-py/' + filename

    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    images = data['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = data['labels']
    imagearray = np.array(images)   #   (10000, 3072)
    labelarray = np.array(labels)   #   (10000,)

    with open('data-cifar10/cifar-10-batches-py/' + 'batches.meta', 'rb') as f:
        metadata = pickle.load(f, encoding='latin1')
    label_names = metadata['label_names']

    return imagearray, labelarray, label_names
