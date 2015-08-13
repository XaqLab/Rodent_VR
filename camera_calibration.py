#!/usr/python

"""
Estimate a camera's radial distortion parameters.  

The calibration image is produced when this script is run with no arguments.
Photograph this image with the camera and pass the filename of the photograph
to this script to estimate the camera's distortion parameters.
"""

import sys
from PIL import Image
from numpy import array, ones, uint8, dot, sin, cos, argwhere, delete
from scipy.optimize import minimize
from scipy import ndimage
import logitech_c525 as webcam
from dome_calibration import find_center_points, sort_points, remove_distortion


DEBUG = True

# define constants
CAMERA_PIXEL_WIDTH = 1280
CAMERA_PIXEL_HEIGHT = 720
BACKGROUND_PIXEL_VALUE = 0
OBJECT_PIXEL_VALUE = 192  # < 255 to prevent saturating the camera


def create_diamond_image(center_pixels, diamond_size,
                         image_width=CAMERA_PIXEL_WIDTH,
                         image_height=CAMERA_PIXEL_HEIGHT):
    """
    Make an image with diamonds centered on center_pixels.  The height and
    width of the diamonds is specified by diamond_size.
    """

    # make a dark background
    pixels = ones([image_height, image_width], dtype=uint8)
    pixels = BACKGROUND_PIXEL_VALUE * pixels

    half_size = (diamond_size - 1)/2

    # add the diamonds
    for center_pixel in center_pixels:
        center_row = center_pixel[0]
        center_col = center_pixel[1]
        diamond_pixels = []
        for row in range(center_row - half_size,
                         center_row + half_size + 1):
            for col in range(center_col - half_size + abs(row - center_row),
                             center_col + half_size + 1 - abs(row - center_row)):
                pixels[row][col] = OBJECT_PIXEL_VALUE

    return Image.fromarray(pixels, mode='L')


def camera_distortion(x, image_centers, photo_centers):
    """
    Calculate the sum of the squared error between the pixels in the camera
    calibration image and the pixels in the distortion-corrected photo of the
    camera calibration image.
    """
    # sort x into meaninful names
    scaling_factor = x[0]
    rotation_angle = x[1]
    translation_x = x[2]
    translation_y = x[3]
    distortion_coefficients = x[4:]

    # Rotate, translate and scale the image to remove the effects of
    # these nuisance variables.
    A = array([[ cos(rotation_angle), -sin(rotation_angle)],
               [ sin(rotation_angle),  cos(rotation_angle)]])
    A = scaling_factor * A
    B = array([translation_x, translation_y])
    centers = [list(A.dot(center) + B) for center in photo_centers]

    # remove radial distortion from the camera calibration photo pixels
    centers = remove_distortion(centers, distortion_coefficients)
    x_photo = array([centers[i][0] for i in range(len(centers))])
    y_photo = array([centers[i][1] for i in range(len(centers))])

    # calculate the sum of the square differences between the image pixel
    # values and the pixel values from the photo
    x_image = array([image_centers[i][0] for i in range(len(image_centers))])
    y_image = array([image_centers[i][1] for i in range(len(image_centers))])

    # for each image point, find the closest photo point to calculate errors
    sum_of_square_errors = 0
    for image_center in image_centers:
        # find minimum error (i.e. closest photo point)
        errors = ((x_photo - image_center[0])**2
                    + (y_photo - image_center[1])**2)
        min_error = min(errors)
        # add minimum error to the running total
        sum_of_square_errors = sum_of_square_errors + min_error
        # remove that photo point
        index = argwhere(errors == min_error)
        delete(x_photo, index)
        delete(y_photo, index)
    
    return sum_of_square_errors

    # this assumes that the points have been sorted so they are in the same
    # order in both the image and the photo
    #value = sum((x_photo - x_image)**2 + (y_photo - y_image)**2)

    #return value


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":
    """
    Define camera_pixels which will be used to generate the camera calibration
    image so we can compensate for the distortion introduced by the camera we
    use for calibration.  The pixels must be listed from left to right and
    then top to bottom so they can be matched with the correct pixel in the
    photo.
    """
    diamond_size = 13 # vertical and horizontal size of the diamonds in pixels
    n = 11.0 # affects row spacing
    m = 15.0 # affects column spacing
    num_rows = webcam.pixel_height
    row_nums = ([num_rows*(0.5 - i/n) - 1 for i in range(int(n/2), 0, -1)] +
                [num_rows*(0.5 + i/n) for i in range(1, int(n/2) + 1)])
    num_cols = webcam.pixel_width
    col_nums = ([num_cols*(0.5 - i/m) - 1 for i in range(int(m/2), 0, -1)] +
                [num_cols*(0.5 + i/m) for i in range(1, int(m/2) + 1)])
    camera_pixels = [[int(row_nums[j]), int(col_nums[i])]
                     for j in range(len(row_nums))
                     for i in range(len(col_nums))]

    if len(sys.argv) == 1:
        """
        No arguments were given so generate the calibration image and save
        it to a file.  The calibration image is photographed with the camera
        to enable compensation for its barrel distortion.  
        """
        camera_image = \
                create_diamond_image(camera_pixels, diamond_size)
        camera_image.save("camera_calibration_image.png")

    elif len(sys.argv) == 2:
        """
        An argument was given so estimate the camera's radial distortion
        parameters.  Treat the argument as the file name of the photograph of
        the calibration image taken with the camera.
        """
        photo_filename = sys.argv[1]

        """
        Find the center pixels of the objects in the camera calibration photo
        and estimate the camera's radial distortion coefficients by minimizing
        the difference between these pixels and camera_pixels.
        """
        photo = Image.open(photo_filename).convert('L')
        photo_centers = find_center_points(photo)
        # sort the center points from left to right and top to bottom
        #photo_centers = sort_points(photo_centers, 10, 14)
        photo.close()

        # convert camera image pixels (row, column) to (x, y) coordinates
        image_centers = [[  pixel[1] - CAMERA_PIXEL_WIDTH/2 + 0.5,
                          -(pixel[0] - CAMERA_PIXEL_HEIGHT/2 + 0.5)]
                         for pixel in camera_pixels]

        x0 = array([1] + [1e-19]*3 + [-1e-18]*3)
        arguments = (image_centers, photo_centers)
        results = minimize(camera_distortion, x0, args=arguments,
                           method='Nelder-Mead')
                           #method='L-BFGS-B')
        distortion_coefficients = results['x'][4:]
        print results
        photo_centers = remove_distortion(photo_centers,
                                          distortion_coefficients)
        # Convert (x, y) center points to (row, column) center pixels
        photo_pixels = [[-center_point[1] + CAMERA_PIXEL_HEIGHT/2 - 0.5,
                         center_point[0] + CAMERA_PIXEL_WIDTH/2 - 0.5]
                        for center_point in photo_centers]
        photo_pixels = [[int(round(pixel[0])), int(round(pixel[1]))]
                        for pixel in photo_pixels]
        undistorted_image = \
                create_diamond_image(photo_pixels, diamond_size)
        undistorted_image.show()


