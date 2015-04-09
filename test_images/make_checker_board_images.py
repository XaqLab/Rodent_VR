#!/usr/local/bin/python
# written for python 2.7.8

"""
Short script to create checker board images for dome projection.
"""

DEBUG = True

from numpy import array, ones, zeros, dstack
from numpy import uint8, mod
from PIL import Image


def create_checker_board(square_size, board_size):
    """
    Make a checker board image with squares that have [rows, cols] of pixels
    as specified by square_size and [rows, cols] of squares specified by 
    board_size.
    """

    image_pixel_height = square_size[0] * board_size[0]
    image_pixel_width = square_size[1] * board_size[1]

    #pixels = zeros([image_pixel_height, image_pixel_width, 3], dtype=uint8)
    pixels = zeros([image_pixel_height, image_pixel_width], dtype=uint8)
    for row in range(image_pixel_height):
        for col in range(image_pixel_width):
            if (mod(row, 2*square_size[0]) < square_size[0]
                and mod(col, 2*square_size[1]) < square_size[1]):
                pixels[row][col] = 255
            elif (mod(row, 2*square_size[0]) >= square_size[0]
                and mod(col, 2*square_size[1]) >= square_size[1]):
                pixels[row][col] = 255

    #return Image.fromarray(pixels, mode='RGB')
    return Image.fromarray(pixels, mode='L')


if __name__ == "__main__":
    # total pixels: 1280 by 720
    #checker_board_image = create_checker_board([20, 20], [36, 64])
    #checker_board_image.save("checker_board_20_by_20.png")
    #checker_board_image = create_checker_board([1, 1], [720, 1280])
    #checker_board_image.save("checker_board_1_by_1.png")
    #checker_board_image = create_checker_board([2, 2], [360, 640])
    #checker_board_image.save("checker_board_2_by_2.png")
    checker_board_image = create_checker_board([128, 128], [4, 4])
    checker_board_image.save("checker_board_4_by_4.png")


