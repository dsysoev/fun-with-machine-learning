
import numpy as np
import matplotlib.pyplot as plt

from data import load_batch


def plot_image(index, images, labels, label_names):
    image, label = images[index, :, :], labels[index]
    image = image.reshape(3, 32, 32).transpose([1, 2, 0])
    plt.imshow(image)
    plt.title("class: {} class name: {}".format(label, label_names[label]))
    plt.show()

if __name__ in '__main__':
    im, il, ln = load_batch()
    for i in range(10):
        plot_image(i, im, il, ln)
