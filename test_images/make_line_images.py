#!/usr/local/bin/python
# written for python 2.7.8

"""
Short script to create horizontal and vertical line images for dome projection.
"""

DEBUG = True

from numpy import array, ones, zeros, dstack
from numpy import uint8, mod
from PIL import Image


def create_vertical_lines(image_size, line_width, line_spacing):
    """
    Make an image of size image_size ([rows, cols]) with vertical lines that
    have a pixel width specified by line_width.
    """

    image_pixel_height = image_size[0]
    image_pixel_width = image_size[1]

    #pixels = zeros([image_pixel_height, image_pixel_width, 3], dtype=uint8)
    pixels = zeros([image_pixel_height, image_pixel_width], dtype=uint8)
    for row in range(image_pixel_height):
        for col in range(image_pixel_width):
            if (mod(col, line_width + line_spacing) < line_width):
                pixels[row][col] = 255

    #return Image.fromarray(pixels, mode='RGB')
    return Image.fromarray(pixels, mode='L')


def create_horizontal_lines(image_size, line_width, line_spacing):
    """
    Make an image of size image_size ([rows, cols]) with horizontal lines that
    have a pixel width specified by line_width.
    """

    image_pixel_height = image_size[0]
    image_pixel_width = image_size[1]

    #pixels = zeros([image_pixel_height, image_pixel_width, 3], dtype=uint8)
    pixels = zeros([image_pixel_height, image_pixel_width], dtype=uint8)
    for col in range(image_pixel_width):
        for row in range(image_pixel_height):
            if (mod(row, line_width + line_spacing) < line_width):
                pixels[row][col] = 255

    #return Image.fromarray(pixels, mode='RGB')
    return Image.fromarray(pixels, mode='L')


if __name__ == "__main__":
    line_image = create_vertical_lines([720, 1280], 16, 616)
    line_image.save("center_line.png")
    #line_image = create_vertical_lines([512, 512], 16, 16)
    #line_image.save("vertical_lines_16.png")
    #line_image = create_horizontal_lines([512, 512], 16, 16)
    #line_image.save("horizontal_lines_16.png")


