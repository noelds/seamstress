import numpy as np


def gradient_magnitude(image, order=1):
    """Return the magnitude of the gradient of the image.

    :type image: ndarray
    :rtype ndarray
    """
    dx, dy = np.zeros_like(image), np.zeros_like(image)
    dx[:, :-order] = np.diff(image, n=order, axis=1)
    dy[:-order, :] = np.diff(image, n=order, axis=0)
    return np.sqrt(dx ** 2 + dy ** 2)
