import logging

from matplotlib import pyplot as plt
from skimage import io

from seamstress.carver import SeamCarver


def save_image(name, image):
    """Save <image> in the current directory as <name>.

    :type name: str
    :type image: ndarray
    :rtype: None
    """
    plt.imshow(image)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(name, bbox_inches='tight')


def example():
    """Run the seam carving example.

    The example runs iterations of seam carving on a sample image, lena.jpg.
    Results are saved to the local directory.
    """
    seam_counts = [10, 50]
    image = io.imread('lena.jpg')
    carver = SeamCarver(image)

    for count in seam_counts:
        carver.find_seams(count)
        save_image('lena_{}_seams.png'.format(count), carver.color())
        save_image('lena_{}_shrunk.png'.format(count), carver.shrink())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example()
