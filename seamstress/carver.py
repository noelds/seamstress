import logging

import numpy as np

from seamstress.energy import gradient_magnitude


class SeamCarver:
    """Return a SeamCarver for a single image. The SeamCarver can be used to find seams,
    visualize seams, and remove seams.
    """

    def __init__(self, image):
        """Initializer.

        :type image: ndarray
        :return: None
        """
        self.image = image
        self.height, self.width = image.shape[0], image.shape[1]
        self.seams = []
        self.is_altered = False

    def find_seams(self, count):
        """Find <count> seams in the image.

        :type count: int
        :rtype: None
        """
        self.seams = []
        temp = np.copy(self.image)
        for i in range(count):
            seam = cheapest_vertical_seam(temp)
            self.seams.append(seam)
            logging.info('Seams found: {i}'.format(i=i + 1))
            temp = delete_seam(temp, seam)
        return self

    def _create_mask(self, inverted=False):
        """Return a boolean mask using found seams in the image.

        :type inverted: bool
        :rtype: ndarray
        """
        if not inverted:
            mask = np.ones((self.height, self.width), dtype=bool)
        else:
            mask = np.zeros((self.height, self.width), dtype=bool)

        for y, seam_layer in enumerate(zip(*self.seams)):
            for i, x in enumerate(seam_layer):
                for j in reversed(range(i)):
                    if seam_layer[j] <= x:
                        # Augment x if a seam previously removed, add one to x.
                        x += 1
                mask[y, x] = inverted
        return mask

    def shrink(self):
        """Return an image that has been reduced in size using found seams.

        :rtype: ndarray
        """
        dup = np.copy(self.image)
        return dup[self._create_mask()].reshape(self.height,
                                                self.width - len(self.seams),
                                                -1)

    def grow(self):
        """Return an image that has been enlarged in size using found seams.

        :rtype: ndarray
        """
        return NotImplemented

    def color(self, rgb=(255, 0, 0)):
        """Color the seam in the image.

        :type rgb: (int, int, int)
        :rtype: ndarray
        """
        duplicate = np.copy(self.image)
        duplicate[self._create_mask(inverted=True)] = rgb
        return duplicate


def cheapest_vertical_seam(image):
    """Return the indices of the seam in <image> with the least energy.

    :type image: ndarray
    :rtype list[int]
    """
    gradient = np.mean(gradient_magnitude(image), axis=2)
    height, width = gradient.shape
    seams = np.zeros(gradient.shape, dtype=int)
    cost = np.zeros(gradient.shape)
    cost[0, :] = gradient[0, :]

    # Build a table of seams.
    for y in range(1, height):
        for x in range(width):
            start_shift = -1 if x > 0 else 0
            stop_shift = 2 if x < width - 1 else 1
            cheapest = x + start_shift + np.argmin(cost[y - 1, x + start_shift:x + stop_shift])
            cost[y, x] = cost[y - 1, cheapest] + gradient[y, x]
            seams[y, x] = cheapest

    # Find the smallest seam from bottom to top.
    crumb = int(np.argmin(cost[-1, :]))
    seam = [crumb]
    for y in range(height - 2, -1, -1):
        crumb = int(seams[y, crumb])
        seam.append(crumb)
    seam.reverse()

    return seam


def delete_seam(image, seam):
    """Delete <seam> from <image>.

    :type image: ndarray
    :type seam: collections.iterable[int]
    :rtype: ndarray
    """
    height, width = image.shape[0], image.shape[1]
    mask = np.ones((height, width), dtype=bool)
    mask[[range(len(seam)), seam]] = False
    return image[mask].reshape(height, width - 1, -1)
