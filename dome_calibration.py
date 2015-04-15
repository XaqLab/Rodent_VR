#!/usr/python

"""
Script for finding the dome display parameters from a picture of an image
containing objects that are centered on known projector pixels.
"""

import sys
from PIL import Image
from numpy import array, ones, uint8, mean, dot, sin, arcsin
from scipy.optimize import minimize
from dome_projection import DomeProjection, flat_display_direction
import webcam

# increase recursion limit for add_pixel_to_object
sys.setrecursionlimit(10000)

DEBUG = False

# define constants
PROJECTOR_PIXEL_WIDTH = 1280
PROJECTOR_PIXEL_HEIGHT = 720
BACKGROUND_PIXEL_VALUE = 0
OBJECT_PIXEL_VALUE = 192


def create_calibration_image(center_pixels, diamond_size):
    """
    Make an image with diamonds centered on center_pixels.  The height and
    width of the diamonds is specified by diamond_size.
    """

    # make a dark background
    pixels = ones([PROJECTOR_PIXEL_HEIGHT, PROJECTOR_PIXEL_WIDTH], dtype=uint8)
    pixels = BACKGROUND_PIXEL_VALUE * pixels

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
                pixels[row][col] = OBJECT_PIXEL_VALUE

    return Image.fromarray(pixels, mode='L')


def find_center_pixels(image_filename):
    """
    This function returns [row, column] coordinates for the pixels on which
    objects in the calibration image are centered.
    """

    # define the pixel threshold used to distinguish objects from the background
    PIXEL_THRESHOLD = (BACKGROUND_PIXEL_VALUE + OBJECT_PIXEL_VALUE) / 2

    # read image from file
    image = Image.open(image_filename).convert('L')
    [image_width, image_height] = image.size
    pixels = array(image)
    image.close()
    
    # find all the object pixels
    object_pixels = []
    for row in range(image_height):
        for column in range(image_width):
            if pixels[row, column] > PIXEL_THRESHOLD:
                object_pixels.append([row, column])
    
    
    def add_pixel_to_object(object_pixels, index, objects):
        """
        Add object_pixels[index] to the last object in the objects list and
        then check the 8 adjacent pixels to see if any of them are among the
        remaining object pixels and add them to the object if they are.
        """
        pixel = object_pixels.pop(index)
        objects[-1].append(pixel)
        
        # check adjacent pixels
        row_above = [[-1, -1], [-1, 0], [-1, 1]]
        beside = [[0, -1], [0, 1]]
        row_below = [[1, -1], [1, 0], [1, 1]]
        offsets = row_above + beside + row_below
        pixel_row = pixel[0]
        pixel_column = pixel[1]
        neighbors = [[pixel_row + dr, pixel_column + dc]
                     for [dr, dc] in offsets]
        for neighbor in neighbors:
            if neighbor in object_pixels:
                add_pixel_to_object(object_pixels,
                                    object_pixels.index(neighbor), objects)
    
    
    # sort object pixels into sets of adjacent pixels (i.e. objects)
    objects = []
    while len(object_pixels) > 0:
        # Start a new object and add the first object pixel to it, recursion
        # will take care of the rest.
        objects.append([])
        add_pixel_to_object(object_pixels, 0, objects)
    
    
    # Avoid counting small groups of pixels from image distortion as objects by
    # finding the number of pixels in the largest object and ignoring objects which
    # have fewer than 10% of the pixels the largest object has.
    largest_object_size = 0
    for obj in objects:
        object_size = len(obj)
        if object_size > largest_object_size:
            largest_object_size = object_size
    
    
    # estimate which pixel is at the center of each object by averaging the row
    # and column values of the pixels
    center_pixels = []
    for obj in objects:
        if len(obj) > 0.1 * largest_object_size:
            object_pixel_array = array(obj)
            center_row = int(round(mean(object_pixel_array[:,0])))
            center_column = int(round(mean(object_pixel_array[:,1])))
            center_pixels.append([center_row, center_column])
    
    if DEBUG:
        """ invert the center pixels so they can be seen """
        for center_pixel in center_pixels:
            pixels[center_pixel[0], center_pixel[1]] = BACKGROUND_PIXEL_VALUE
        debug_image = Image.fromarray(pixels, mode = 'L')
        debug_image.show()
    
    return center_pixels


def calc_projector_images(y, z, theta, vertical_offset):
    """
    Calculate the two projector_image parameters that the dome class requires
    from a smaller set of parameters that are more parameter estimation
    friendly. The location of the projector's focal point is given by y and z.
    Theta is half the angle between lines from the focal point to the left and
    right sides of the image.  The lens offset of the projector is described by
    vertical_offset.
    """
    # distance to first image, chosen to match measurements
    y1 = 0.436
    # calculate x from theta and the distance between the focal point and image
    x1 = (y - y1) * sin(theta)
    # calculate z by assuming a 16:9 aspect ratio 
    z1_low = vertical_offset
    z1_high = z1_low + 2 * 9.0/16.0 * x1
    image1 = [[ -x1,  y1,  z1_high ],
              [  x1,  y1,  z1_high ],
              [  x1,  y1,  z1_low ],
              [ -x1,  y1,  z1_low ]]

    # do it again for image2
    y2 = 0.265
    x2 = (y - y2) * sin(theta)
    slope = (vertical_offset - z) / (y - y1)
    z2_low = slope * (y - y2)
    z2_high = z2_low + 2 * 9.0/16.0 * x2
    image2 = [[ -x2,  y2,  z2_high ],
              [  x2,  y2,  z2_high ],
              [  x2,  y2,  z2_low ],
              [ -x2,  y2,  z2_low ]]
    
    return [image1, image2]


def calc_frustum_parameters(image1, image2):
    """
    Inverse of calc_projector_images.
    """
    # this one is easy
    vertical_offset = image1[2][2]

    # calculate theta
    x1 = image1[0][0]
    x2 = image2[0][0]
    y1 = image1[0][1]
    y2 = image2[0][1]
    theta = arcsin((x2 - x1) / (y2 - y1))

    # calculate y
    y = y1 - x1 / sin(theta)

    # calculate z
    z2_low = image2[2][2]
    z = vertical_offset - z2_low * (y - y1)/(y - y2)
    
    return [y, z, theta, vertical_offset]


def minimization_function(x, image_pixels, measured_directions):
    """
    Calculate the negative sum of the dot products between the measured
    directions and the calculated directions.  Minimizing this function is
    equivalent to maximizing the sum of the dot products.
    """
    # decode entries in x into meaningful names
    projector_y = x[0]
    projector_z = x[1]
    theta = x[2]
    vertical_offset = x[3]
    projector_images = calc_projector_images(projector_y, projector_z, theta,
                                             vertical_offset)
    #mirror_radius = x[4]
    #dome_y = x[5]
    #dome_z = x[6]
    #dome_radius = x[7]
    #animal_y = x[8]
    #animal_z = x[9]

    mirror_radius = 0.215
    dome_y = 0.138
    dome_z = 0.309
    dome_radius = 0.603
    animal_y = 0.06
    animal_z = 0.61

    # setup the parameters dictionary
    parameters = dict(screen_height = [webcam.screen_height],
                      screen_width = [webcam.screen_width],
                      distance_to_screen = [webcam.distance_to_screen],
                      pitch = [30],
                      yaw = [0],
                      roll = [0],
                      image_pixel_width = [1280],
                      image_pixel_height = [720],
                      projector_pixel_width = 1280,
                      projector_pixel_height = 720,
                      first_projector_image = projector_images[0],
                      second_projector_image = projector_images[1],
                      mirror_radius = mirror_radius,
                      dome_center = [0, dome_y, dome_z],
                      dome_radius = dome_radius,
                      animal_position = [0, animal_y, animal_z])

    # Calculate the directions to the center pixels in the calibration
    # image using these parameters.
    dome = DomeProjection(**parameters)
    calculated_directions = []
    for image_pixel in image_pixels:
        [row, col] = image_pixel
        calculated_direction = dome._dome_display_direction(row, col)[1]
        calculated_directions.append(calculated_direction)
    if DEBUG:
        print "Calculated directions:", calculated_directions

    return -sum([measured_directions[i].dot(calculated_directions[i])
                 for i in range(len(measured_directions))])


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":
    # Define image_pixels here because they are needed for both image
    # generation and parameter estimation.
    diamond_size = 19
    image_pixels = [[500, 540], [500, 740]]
    if len(sys.argv) == 1:
        """
        No arguments given so generate the calibration image and save it to a
        file.  This image contains objects centered on known projector pixels.
        Project the image on the dome display and take a picture, then pass
        the filename of this picture as an argument to this program.
        """
        calibration_image = \
                create_calibration_image(image_pixels, diamond_size)
        calibration_image.save("dome_calibration.png")
    else:
        """
        At least one argument given, treat the first argument as the file name
        of the calibration picture.  Find the center pixels of the objects in
        the picture, calculate the directions from the camera's focal point to
        these pixels and then search the parameter space for parameter values
        that result in calculated directions for these pixels that match these
        measured directions.
        """
        calibration_photo = sys.argv[1]

        # Find the center pixels of the objects in the photograph of the
        # calibration image projected onto the dome.
        photo_pixels = find_center_pixels(calibration_photo)

        # Calculate the directions to photo_pixels using the camera's FoV.
        measured_directions = []
        for photo_pixel in photo_pixels:
            [row, column] = photo_pixel
            arguments = [row, column, webcam.screen_height,
                         webcam.screen_width, webcam.pixel_height,
                         webcam.pixel_width, webcam.distance_to_screen]
            measured_directions.append(flat_display_direction(*arguments))

        if DEBUG:
            print "Image pixels:", image_pixels
            print "Photo pixels:", photo_pixels
            print "Measured directions:", measured_directions


        # Search the parameter space to find values that maximize the dot
        # products between measured_directions and calculated directions.

        """
        Note: there is currently nothing that guarantees that image_pixels and
        photo_pixels are in the same order!  If they are not then the results
        will not be useful.
        """

        first_projector_image = [[-0.080, 0.436, 0.137],
                                 [0.080, 0.436, 0.137],
                                 [0.080, 0.436, 0.043],
                                 [-0.080, 0.436, 0.043]]
        second_projector_image = [[-0.115, 0.265, 0.186],
                                  [0.115, 0.265, 0.186],
                                  [0.115, 0.265, 0.054],
                                  [-0.115, 0.265, 0.054]]
        mirror_radius = 0.215
        dome_center = [0, 0.138, 0.309]
        dome_radius = 0.603
        animal_position = [0, 0.06, 0.61]
        
        x0 = (calc_frustum_parameters(first_projector_image,
                                     second_projector_image) + 
              [mirror_radius, dome_center[1], dome_center[2], dome_radius,
               animal_position[1], animal_position[2]])
        #import pdb; pdb.set_trace()
        f = minimization_function
        arguments = (image_pixels, measured_directions)
        results = minimize(f, x0, args=arguments)
        print results
        projector_images = calc_projector_images(*results['x'][0:4])
        for image in projector_images:
            for row in image:
                print row
        print "Mirror radius:", results['x'][4]
        print "Dome y-coordinate:", results['x'][5]
        print "Dome z-coordinate:", results['x'][6]
        print "Dome radius:", results['x'][7]
        print "Animal y-coordinate:", results['x'][8]
        print "Animal z-coordinate:", results['x'][9]


