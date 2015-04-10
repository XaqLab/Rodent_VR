#!/usr/local/bin/python
# written for python 2.7.8

"""
Short script to create an image containing diamonds centered on specific pixels
for dome projection calibration.
"""

DEBUG = True

from numpy import array, ones, dstack
from numpy import uint8, mod
from PIL import Image


def create_diamond_image(center_pixels, diamond_size):
    """
    Make an image with diamonds centered on center_pixels.  The height and
    width of the diamonds is specified by diamond_size.
    """

    IMAGE_PIXEL_WIDTH = 1280
    IMAGE_PIXEL_HEIGHT = 720
    DARK_PIXEL_VALUE = 0
    LIGHT_PIXEL_VALUE = 192

    # make a dark background
    pixels = ones([IMAGE_PIXEL_HEIGHT, IMAGE_PIXEL_WIDTH], dtype=uint8)
    pixels = DARK_PIXEL_VALUE * pixels

    half_size = (diamond_size - 1)/2

    # add the diamonds
    for center_pixel in center_pixels:
        center_row = center_pixel[0]
        center_col = center_pixel[1]
        diamond_pixels = []
        for row in range(center_row - half_size,
                         center_row + half_size):
            for col in range(center_col - half_size + abs(row - center_row),
                             center_col + half_size - abs(row - center_row)):
                pixels[row][col] = LIGHT_PIXEL_VALUE

    return Image.fromarray(pixels, mode='L')


if __name__ == "__main__":
    diamond_size = 19
    center_pixels = [[500, 540], [500, 740]]
    checker_board_image = create_diamond_image(center_pixels, diamond_size)
    checker_board_image.save("diamonds.png")


