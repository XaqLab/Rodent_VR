#!/usr/python

"""
Script for finding the dome display parameters from a picture of an image
containing objects that are centered on known projector pixels.
"""

import sys
from PIL import Image
from numpy import array, ones, uint8, mean, dot, tan, arctan
from numpy import histogram, diff, sign
from scipy.optimize import minimize
from dome_projection import DomeProjection, flat_display_direction
import webcam

# increase recursion limit for add_pixel_to_object
sys.setrecursionlimit(10000)

DEBUG = True

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

    # read image from file
    image = Image.open(image_filename).convert('L')
    [image_width, image_height] = image.size
    pixels = array(image)
    image.close()
    
    # define the pixel threshold used to distinguish objects from the background
    PIXEL_THRESHOLD = (int(pixels.max()) + int(pixels.min())) / 2

    # find all the object pixels
    object_pixels = []
    for row in range(image_height):
        for column in range(image_width):
            if pixels[row, column] > PIXEL_THRESHOLD:
                object_pixels.append([row, column])

    # build histograms of object pixels row and column values
    row_values = [object_pixels[i][0] for i in range(len(object_pixels))]
    row_counts = histogram(row_values, range(image_height + 1))[0]
    col_values = [object_pixels[i][1] for i in range(len(object_pixels))]
    col_counts = histogram(col_values, range(image_width + 1))[0]

    # The peaks in this multimodal distribution are separated by intervals
    # containing all zeros.  Find the center of these zero intervals and use
    # them as thresholds for separating objects.

    # Find the row thresholds.
    start = 0
    end = 0
    row_thresholds = []
    #import pdb; pdb.set_trace()
    while end < len(row_counts):
        if row_counts[start] == 0:
            if row_counts[end] != 0:
                # end of interval, add center to thresholds
                row_thresholds.append((start + end)/2)
                start = end
        else:
            # looking for the beginning of an interval
            start = start + 1
        end = end + 1
    # throw away first threshold when there are no objects
    # on the edge of the image
    if row_counts[0] == 0:
        row_thresholds = row_thresholds[1:]

    # Find the column thresholds.
    start = 0
    end = 0
    col_thresholds = []
    while end < len(col_counts):
        if col_counts[start] == 0:
            if col_counts[end] != 0:
                # end of interval, add center to thresholds
                col_thresholds.append((start + end)/2)
                start = end
        else:
            # looking for the beginning of an interval
            start = start + 1
        end = end + 1
    # throw away first threshold when there are no objects
    # on the edge of the image
    if col_counts[0] == 0:
        col_thresholds = col_thresholds[1:]
    
    if DEBUG:
        for row_threshold in row_thresholds:
            for col in range(1280):
                pixels[row_threshold, col] = OBJECT_PIXEL_VALUE

        for col_threshold in col_thresholds:
            for row in range(720):
                pixels[row, col_threshold] = OBJECT_PIXEL_VALUE

        print row_thresholds
        print col_thresholds
        debug_image = Image.fromarray(pixels, mode = 'L')
        debug_image.show()
    

    # Sort object pixels into objects, reverse the order of the column
    # thresholds so when we're done the objects are sorted from right to left,
    # top to bottom as needed to match the correct image pixel.
    row_thresholds.append(image_height - 1)
    col_thresholds.reverse()
    col_thresholds.append(0)
    objects = []
    for row_threshold in row_thresholds:
        for col_threshold in col_thresholds:
            # start a new object
            objects.append([])
            i = len(object_pixels) - 1
            while i > 0:
                if (object_pixels[i][0] <= row_threshold and
                    object_pixels[i][1] >= col_threshold):
                    object_pixel = object_pixels.pop(i)
                    objects[-1].append(object_pixel)
                i = i - 1


    # estimate which pixel is at the center of each object by averaging the row
    # and column values of the pixels
    center_pixels = []
    for obj in objects:
        object_pixel_array = array(obj)
        center_row = int(round(mean(object_pixel_array[:,0])))
        center_column = int(round(mean(object_pixel_array[:,1])))
        center_pixels.append([center_row, center_column])
    
    if DEBUG:
    #if False:
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
    x1 = (y - y1) * tan(theta)
    # calculate z by assuming a 16:9 aspect ratio 
    z1_low = vertical_offset
    z1_high = z1_low + 2 * 9.0/16.0 * x1
    image1 = [[ -x1,  y1,  z1_high ],
              [  x1,  y1,  z1_high ],
              [  x1,  y1,  z1_low ],
              [ -x1,  y1,  z1_low ]]

    # do it again for image2
    y2 = 0.265
    x2 = (y - y2) * tan(theta)
    slope = (vertical_offset - z) / (y - y1)
    z2_low = z + slope * (y - y2)
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
    x1 = image1[1][0]
    x2 = image2[1][0]
    y1 = image1[1][1]
    y2 = image2[1][1]
    theta = arctan((x2 - x1) / (y1 - y2))

    # calculate y
    y = y1 + x1 / tan(theta)

    # calculate z
    z2_low = image2[2][2]
    slope = (vertical_offset - z2_low) / (y1 - y2)
    z = vertical_offset + slope * (y - y1)
    
    return [y, z, theta, vertical_offset]


def calc_webcam_FoV(theta):
    distance_to_screen = 3.0
    screen_width = 2 * distance_to_screen * tan(theta)
    screen_height = screen_width / (1280.0 / 720.0)
    return [screen_height, screen_width, distance_to_screen]


def calc_webcam_theta(screen_height, screen_width, distance_to_screen):
    theta = arctan(0.5 * screen_width / distance_to_screen)
    return theta


def minimization_function(x, image_pixels, photo_pixels, webcam_theta):
    """
    Calculate the negative sum of the dot products between the measured
    directions and the calculated directions.  Minimizing this function is
    equivalent to maximizing the sum of the dot products.
    """
    # decode entries in x into meaningful names
    projector_y = x[0]
    projector_z = x[1]
    projector_theta = x[2]
    vertical_offset = x[3]
    projector_images = calc_projector_images(projector_y, projector_z,
                                             projector_theta, vertical_offset)
    mirror_radius = x[4]
    dome_y = x[5]
    dome_z = x[6]
    dome_radius = x[7]
    animal_y = x[8]
    animal_z = x[9]

    # Calculate the directions to photo_pixels using the camera's estimated
    # field of view.
    [height, width, distance] = calc_webcam_FoV(webcam_theta)
    measured_directions = []
    for photo_pixel in photo_pixels:
        [row, column] = photo_pixel
        arguments = [row, column, height, width, webcam.pixel_height,
                    webcam.pixel_width, distance]
        measured_directions.append(flat_display_direction(*arguments))

    # setup the parameters dictionary
    parameters = dict(screen_height = [height],
                      screen_width = [width],
                      distance_to_screen = [distance],
                      pitch = [0],
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
    #if DEBUG:
    if False:
        print "Measured directions:", measured_directions
        print "Calculated directions:", calculated_directions

    value = -sum([measured_directions[i].dot(calculated_directions[i])
                 for i in range(len(measured_directions))])
    #print value,
    return value


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":
    # Define image_pixels here because they are needed for both image
    # generation and parameter estimation.  They must be listed from left to
    # right and then top to bottom so they can be matched with the correct
    # direction.
    diamond_size = 13
    #image_pixels = [[500, 540], [500, 740]]
    image_pixels = [[500, 512], [500, 563], [500, 614],
                    [500, 665], [500, 716], [500, 767],
                    [545, 512], [545, 563], [545, 614],
                    [545, 665], [545, 716], [545, 767],
                    [590, 512], [590, 563], [590, 614],
                    [590, 665], [590, 716], [590, 767],
                    [635, 512], [635, 563], [635, 614],
                    [635, 665], [635, 716], [635, 767]]
    if len(sys.argv) == 1:
        """
        No arguments given so generate the calibration image and save it to a
        file.  This image contains objects centered on known projector pixels.
        Project the image on the dome display and take a picture, then pass
        the filename of this picture as an argument to this program.
        """
        calibration_image = \
                create_calibration_image(image_pixels, diamond_size)
        calibration_image.save("calibration_image.png")
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

        #if DEBUG:
        print "Image pixels:", image_pixels
        print "Photo pixels:", photo_pixels

        # Search the parameter space to find values that maximize the dot
        # products between measured_directions and calculated directions.

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
              [mirror_radius,
               dome_center[1],
               dome_center[2],
               dome_radius,
               animal_position[1],
               animal_position[2]])
        parameter_bounds = [(None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None)]
        f = minimization_function
        test_image = Image.open("test_images/WebCameras/Image42.jpg")
        for webcam_theta in [0.5]:
            arguments = (image_pixels, photo_pixels, webcam_theta)
            results = minimize(f, x0, args=arguments, method='L-BFGS-B',
                               bounds=parameter_bounds)
            print results
            projector_y = results['x'][0]
            projector_z = results['x'][1]
            projector_theta = results['x'][2]
            vertical_offset = results['x'][3]
            projector_images = calc_projector_images(projector_y, projector_z,
                                             projector_theta, vertical_offset)
            mirror_radius = results['x'][4]
            dome_y = results['x'][5]
            dome_z = results['x'][6]
            dome_radius = results['x'][7]
            animal_y = results['x'][8]
            animal_z = results['x'][9]

            for image in projector_images:
                for row in image:
                    print row
            print "Mirror radius:", mirror_radius
            print "Dome y-coordinate:", dome_y
            print "Dome z-coordinate:", dome_z
            print "Dome radius:", dome_radius
            print "Animal y-coordinate:", animal_y
            print "Animal z-coordinate:", animal_z
            print "Webcam theta:", webcam_theta
            print "screen_height, screen_width, distance_to_screen"
            [screen_height, screen_width, distance_to_screen] = \
            calc_webcam_FoV(webcam_theta)
            print screen_height, screen_width, distance_to_screen
            dome = DomeProjection(screen_height=[screen_height],
                                  screen_width=[screen_width],
                                  distance_to_screen=[distance_to_screen],
                                  pitch = [0],
                                  yaw = [0],
                                  roll = [0],
                                  image_pixel_width = [1280],
                                  image_pixel_height = [720],
                                  first_projector_image=projector_images[0],
                                  second_projector_image=projector_images[1],
                                  mirror_radius=mirror_radius,
                                  dome_center=[0, dome_y, dome_z],
                                  animal_position = [0, animal_y, animal_z])
            warped_image = dome.warp_image_for_dome([test_image])
            warped_image.save("warped_Image42_" + str(webcam_theta) + ".jpg",
                              "jpeg")
            




