
import numpy as np
import matplotlib.pyplot as plt


def plot_image(index, test_img, pred_img):
    # original on the left
    # predicted on the right
    union = np.concatenate([
        test_img[index, :].reshape(28, 28),
        pred_img[index, :].reshape(28, 28)
        ], axis=1)
    plt.imshow(union, cmap='gray')
    plt.show()


test_inputs = np.loadtxt('original.txt', dtype=float)
pred_inputs = np.loadtxt('prediction.txt', dtype=float)

num_samples = 10
indexes = np.random.choice(test_inputs.shape[0], num_samples)
for index in indexes:
    plot_image(index, test_inputs, pred_inputs)
