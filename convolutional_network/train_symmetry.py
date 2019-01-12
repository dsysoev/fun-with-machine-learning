
import numpy as np


def encode_labels(y, k):
    """
    function convert labels to one hot vector
    """
    onehot = np.zeros((y.shape[0], k))
    for idx, val in enumerate(y):
        onehot[idx, val] = 1.0
    return onehot

# set random seed
np.random.seed(10)
num_samples = 3
num_layers = 2
img_w, img_h = 4, 4

# create inputs from random
inputs = (np.random.randn(*(num_samples, num_layers, img_h, img_w)) > 1.).astype(int)
print(inputs)

# target
# sum of elemets on left part of image more then right part
left_part = img_w // 2
targets = np.zeros((num_samples, 1))
for k in range(inputs.shape[0]):
    a, b = np.sum(inputs[k,:,:,:left_part]), np.sum(inputs[k,:,:,left_part:])
    targets[k, :] = int(a >= b)

# convert target to one hot
targets = encode_labels(targets.astype(int), 2)
print(targets)
