#!/usr/local/bin/python
# written for python 2.7.8

"""
Short script to create grid of diamonds image for dome projection calibration.
"""

DEBUG = True

from numpy import array, ones, zeros, dstack
from numpy import uint8, mod
from PIL import Image


def create_diamond_grid(diamond_spacing, diamond_size):
    """
    Make a checker board image with squares that have [rows, cols] of pixels
    as specified by square_size and [rows, cols] of squares specified by 
    board_size.
    """
    image_pixel_width = 1280
    image_pixel_height = 720

    #pixels = zeros([image_pixel_height, image_pixel_width, 3], dtype=uint8)
    pixels = zeros([image_pixel_height, image_pixel_width], dtype=uint8)

    def make_row_white(row):
        """ Sub-function for making blank rows """
        for col in range(image_pixel_width):
            pixels[row][col] = 255

    row = 0
    col = 0

    while row < image_pixel_height - diamond_spacing:
        # white space rows
        for i in range(row, row + diamond_spacing - diamond_size):
            make_row_white(row)
            row = row + 1
    
        # seed row
        for col in range(image_pixel_width):
            if (col + diamond_spacing/2.0) % diamond_spacing == 0:
                pixels[row][col] = 0
            else:
                pixels[row][col] = 255
        row = row + 1
    
        # grow from seed
        for i in range((diamond_size + 1)/2 - 1):
            for col in range(image_pixel_width):
                if (pixels[row - 1][col] == 0):
                    # pixel 1 row above is black
                    pixels[row][col] = 0
                elif (col > 0 and pixels[row - 1][col - 1] == 0):
                    # pixel 1 row above and 1 row left is black
                    pixels[row][col] = 0
                elif (col < image_pixel_width - 1 and pixels[row - 1][col + 1] == 0):
                    # pixel 1 row above and 1 row right is black
                    pixels[row][col] = 0
                else:
                    pixels[row][col] = 255
            row = row + 1
    
        # shrink from max width
        for i in range((diamond_size + 1)/2 - 1):
            for col in range(image_pixel_width):
                if (col > 0 and col < image_pixel_width - 1 and
                    pixels[row - 1][col - 1] == 0 and
                    pixels[row - 1][col] == 0 and
                    pixels[row - 1][col + 1] == 0):
                    # 3 pixels above are black
                    pixels[row][col] = 0
                else:
                    pixels[row][col] = 255
            row = row + 1
    

    while row < image_pixel_height:
        # finish the image with white space rows
        make_row_white(row)
        row = row + 1


    #return Image.fromarray(pixels, mode='RGB')
    return Image.fromarray(pixels, mode='L')


if __name__ == "__main__":
    checker_board_image = create_diamond_grid(60, 19)
    checker_board_image.save("diamonds.png")


