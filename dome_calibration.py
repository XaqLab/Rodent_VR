#!/usr/python

"""
Estimate the dome projection parameters.  

Two calibration images are produced when this script is run with no arguments.
The first calibration image is photographed with the camera to estimate its
distortion parameters.  The second image is projected onto the dome and
photographed with the camera for parameter estimation.

When these two photographs are passed to this script as arguments, it returns
the estimated parameters.
"""

import sys
from PIL import Image
from numpy import array, ones, uint8, mean, dot, tan, arctan, sin, cos, linalg
from numpy import histogram, diff, sign
from scipy.optimize import minimize, fsolve
from dome_projection import DomeProjection, flat_display_direction
import webcam


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
                         center_row + half_size + 1):
            for col in range(center_col - half_size + abs(row - center_row),
                             center_col + half_size + 1 - abs(row - center_row)):
                pixels[row][col] = OBJECT_PIXEL_VALUE

    return Image.fromarray(pixels, mode='L')


def find_center_pixels(image_filename):
    """
    This function returns [row, column] coordinates for the pixels on which
    objects in the calibration image are centered. The pixels are ordered from
    left to right and then top to bottom.
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

        debug_image = Image.fromarray(pixels, mode = 'L')
        debug_image.show()
    

    # Sort object pixels into objects when we're done the objects are ordered
    # from left to right and then top to bottom.
    row_thresholds.append(image_height - 1)
    col_thresholds.append(image_width - 1)
    objects = []
    for row_threshold in row_thresholds:
        for col_threshold in col_thresholds:
            # start a new object
            objects.append([])
            i = len(object_pixels) - 1
            while i > 0:
                if (object_pixels[i][0] <= row_threshold and
                    object_pixels[i][1] <= col_threshold):
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


def remove_distortion(pixels, coefficients):
    """
    Map distorted pixels to undistorted pixels using a simplified
    Brown-Conrady model.
    """
    # convert pixels (row, col) to points (x, y)
    points = [[pixel[1], -pixel[0]] for pixel in pixels]

    # shift the image so it's centered at (0, 0)
    x_center = webcam.pixel_width/2 - 0.5
    y_center = -(webcam.pixel_height/2 - 0.5)
    points = [[point[0] - x_center, point[1] - y_center] for point in points]

    # calculate the undistorted x and y values from the distorted values
    x = array([points[i][0] for i in range(len(points))])
    y = array([points[i][1] for i in range(len(points))])
    K = coefficients
    r2 = x**2 + y**2 
    x_undistorted = x*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)
    y_undistorted = y*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)

    # convert (x, y) points back to (row, col) pixels and shift the image back
    # to its original location
    rows = [int(round(-(i + y_center))) for i in y_undistorted]
    cols = [int(round(i + x_center)) for i in x_undistorted]

    return [[rows[i], cols[i]] for i in range(len(pixels))]


def add_distortion(pixels, coefficients):
    """
    Map undistorted pixels to distorted pixels using a simplified
    Brown-Conrady model.
    """
    # convert pixels (row, col) to points (x, y)
    points = [[pixel[1], -pixel[0]] for pixel in pixels]

    # shift the image so it's centered at (0, 0)
    x_center = webcam.pixel_width/2 - 0.5
    y_center = -(webcam.pixel_height/2 - 0.5)
    points = [[point[0] - x_center, point[1] - y_center] for point in points]

    # calculate the distorted x and y values from the undistorted values
    x_undistorted = array([points[i][0] for i in range(len(points))])
    y_undistorted = array([points[i][1] for i in range(len(points))])
    r2 = x_undistorted**2 + y_undistorted**2
    K = coefficients
    x_distorted = x_undistorted*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)
    y_distorted = y_undistorted*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)

    # convert (x, y) points back to (row, col) pixels and shift the image back
    # to its original location
    rows = [int(round(-(i + y_center))) for i in y_distorted]
    cols = [int(round(i + x_center)) for i in x_distorted]

    return [[rows[i], cols[i]] for i in range(len(pixels))]


def camera_cal_debug(x, image_pixels, photo_pixels):
    
    # sort x into meaninful names
    scaling_factor = x[0]
    rotation_angle = x[1]
    translation_x = x[2]
    translation_y = x[3]
    distortion_coefficients = x[4:]

    # convert image pixels (row, col) to points (x, y)
    points = [[pixel[1], -pixel[0]] for pixel in photo_pixels]

    # shift the image so it's centered at (0, 0)
    x_center = webcam.pixel_width/2 - 0.5
    y_center = webcam.pixel_height/2 - 0.5
    points = [[point[0] - x_center, point[1] + y_center] for point in points]

    # Rotate, translate and scale the image to remove the effects of
    # these nuisance variables.
    A = array([[ cos(rotation_angle), -sin(rotation_angle)],
               [ sin(rotation_angle),  cos(rotation_angle)]])
    A = scaling_factor * A
    B = array([translation_x, translation_y])
    points = [list(A.dot(point) + B) for point in points]

    # convert (x, y) points back to (row, col) pixels and shift the image back
    # to its original location
    pixels = [[-point[1] + y_center, point[0] + x_center] for point in points]

    pixels2 = [[int(round(pixel[0])), int(round(pixel[1]))] for pixel in pixels]
    create_calibration_image(pixels2, 13).show()

    # remove radial distortion from the camera calibration photo pixels
    pixels = remove_distortion(pixels, distortion_coefficients)
    x_photo = array([pixels[i][1] for i in range(len(pixels))])
    y_photo = array([-pixels[i][0] for i in range(len(pixels))])

    print "\nCamera calibration photo pixels before and after distortion removal"
    for i in range(len(pixels)):
        print "%3s" % str(pixels2[i][0] - pixels[i][0]) + ",",
        print "%3s" % str(pixels2[i][1] - pixels[i][1]) + "  ",
        if i % 14 == 13:
            print
    print
    create_calibration_image(pixels, 13).show()

    return


def camera_distortion(x, image_pixels, photo_pixels):
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

    # convert image pixels (row, col) to points (x, y)
    points = [[pixel[1], -pixel[0]] for pixel in photo_pixels]

    # shift the image so it's centered at (0, 0)
    x_center = webcam.pixel_width/2 - 0.5
    y_center = webcam.pixel_height/2 - 0.5
    points = [[point[0] - x_center, point[1] + y_center] for point in points]

    # Rotate, translate and scale the image to remove the effects of
    # these nuisance variables.
    A = array([[ cos(rotation_angle), -sin(rotation_angle)],
               [ sin(rotation_angle),  cos(rotation_angle)]])
    A = scaling_factor * A
    B = array([translation_x, translation_y])
    points = [list(A.dot(point) + B) for point in points]

    # convert (x, y) points back to (row, col) pixels and shift the image back
    # to its original location
    pixels = [[-point[1] + y_center, point[0] + x_center] for point in points]

    # remove radial distortion from the camera calibration photo pixels
    pixels = remove_distortion(pixels, distortion_coefficients)
    x_photo = array([pixels[i][1] for i in range(len(pixels))])
    y_photo = array([-pixels[i][0] for i in range(len(pixels))])

    # calculate the sum of the square differences between the distorted pixel
    # values and the pixel values from the photo
    x_image = array([image_pixels[i][1] for i in range(len(image_pixels))])
    y_image = array([-image_pixels[i][0] for i in range(len(image_pixels))])
    value = sum((x_photo - x_image)**2 + (y_photo - y_image)**2)

    return value


def dome_distortion(x, image_pixels, photo_pixels, webcam_theta):
    """
    Calculate the sum of the square differences between measured and calculated
    directions.
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

    #import pdb; pdb.set_trace()
    value = sum([linalg.norm(measured_directions[i] - calculated_directions[i])
                             for i in range(len(measured_directions))])
    #value = -sum([measured_directions[i].dot(calculated_directions[i])
                 #for i in range(len(measured_directions))])
    #print value,
    return value


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":
    """
    This is the vertical and horizontal pixel size of the diamonds in the
    calibration images.
    """
    diamond_size = 13

    """
    Define camera_pixels which will be used to generate the camera calibration
    image so we can compensate for the distortion introduced by the camera we
    use for calibration.  The pixels must be listed from left to right and
    then top to bottom so they can be matched with the correct pixel in the
    photo.
    """
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

    """
    Define dome_pixels for generation of the dome calibration image. This image
    is used to estimate the dome projection geometry.  Because there is a
    mirror involved, these pixels must be listed from right to left and then
    top to bottom so they can be matched with the correct pixels in the photo.
    """
    dome_pixels = [[500, 767], [500, 716], [500, 665],
                   [500, 614], [500, 563], [500, 512],
                   [545, 767], [545, 716], [545, 665],
                   [545, 614], [545, 563], [545, 512], 
                   [590, 767], [590, 716], [590, 665],
                   [590, 614], [590, 563], [590, 512],
                   [635, 767], [635, 716], [635, 665],
                   [635, 614], [635, 563], [635, 512]]
    if len(sys.argv) == 1:
        """
        No arguments were given so generate the calibration images and save
        them to files.  These images contain objects centered on known
        projector pixels.  
        """

        """
        The camera calibration image is photographed with the camera to enable
        compensation for its barrel distortion.  
        """
        camera_image = \
                create_calibration_image(camera_pixels, diamond_size)
        camera_image.save("camera_calibration_image.png")

        """
        The dome calibration image is projected on the dome and photographed
        with the camera in order to estimate the dome projection parameters.
        """
        dome_image = \
                create_calibration_image(dome_pixels, diamond_size)
        dome_image.save("dome_calibration_image.png")

    elif len(sys.argv) == 3:
        """
        Two arguments were given so estimate the dome projection parameters.
        Treat the first argument as the file name of the camera calibration
        photo and the second as the file name of the dome calibration photo.
        """
        camera_photo = sys.argv[1]
        dome_photo = sys.argv[2]

        """
        Find the center pixels of the objects in the camera calibration photo
        and estimate the camera's radial distortion coefficients by minimizing
        the difference between these pixels and camera_pixels.
        """
        camera_photo_pixels = find_center_pixels(camera_photo)

        x0 = array([1] + [1e-19]*3 + [-1e-18]*3)
        arguments = (camera_pixels, camera_photo_pixels)
        results = minimize(camera_distortion, x0, args=arguments,
                           method='Nelder-Mead')
                           #method='L-BFGS-B')
        distortion_coefficients = results['x'][4:]
        print results

        # Debug only
        x = results['x']
        camera_cal_debug(x, camera_pixels, camera_photo_pixels)
        #print "Camera calibration photo pixels\n", camera_photo_pixels
        #create_calibration_image(photo_pixels, 13).show()

        """
        Find the center pixels of the objects in the dome calibration photo
        and remove distortion introduced by the camera.  Then calculate the
        directions from the camera's focal point to these pixels and search
        the parameter space for parameter values that produce calculated
        directions for these pixels that match the measured directions.
        """
        # Find the center pixels of the objects in the photograph of the
        # calibration image projected onto the dome.
        dome_photo_pixels = find_center_pixels(dome_photo)

        if DEBUG:
            print "Dome image pixels:"
            print dome_pixels
            print "Dome photo pixels with distortion:"
            print dome_photo_pixels

        # Remove the camera's distortion
        dome_photo_pixels = remove_distortion(dome_photo_pixels,
                                              distortion_coefficients)

        if DEBUG:
            print "Dome photo pixels without distortion:"
            print dome_photo_pixels

        """
        Search the parameter space to find values that maximize the dot
        products between measured_directions and calculated directions.
        """
        # Setup initial values of the parameters
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

        # Make sure mirror radius and dome radius are > 0
        parameter_bounds = [(None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (0, None),
                            (None, None),
                            (None, None),
                            (0, None),
                            (None, None),
                            (None, None)]

        # Find the webcam's field of view by guessing some values and warp
        # a test image using each value and see which one looks right.
        #test_image = Image.open("test_images/WebCameras/Image42.jpg")
        test_image = Image.open("test_images/512_by_512/vertical_lines_16.png")
        for webcam_theta in [0.5]:
            # Estimate parameter values by minimizing the difference between
            # the measured and calculated directions.
            arguments = (dome_pixels, dome_photo_pixels, webcam_theta)
            results = minimize(dome_distortion, x0, args=arguments,
                               method='L-BFGS-B', bounds=parameter_bounds)
            print results

            # Sort results into meaningful parameter names
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

            # Print out the estimated parameter values
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

            # Instantiate DomeProjection class with estimated parameters
                                  #screen_height=[screen_height],
                                  #screen_width=[screen_width],
                                  #distance_to_screen=[distance_to_screen],
            dome = DomeProjection(
                                  screen_height = [1],
                                  screen_width = [1],
                                  distance_to_screen = [0.5],
                                  pitch = [30],
                                  yaw = [0],
                                  roll = [0],
                                  image_pixel_width = [512],
                                  image_pixel_height = [512],
                                  first_projector_image=projector_images[0],
                                  second_projector_image=projector_images[1],
                                  mirror_radius=mirror_radius,
                                  dome_center=[0, dome_y, dome_z],
                                  animal_position = [0, animal_y, animal_z])

            # Warp a test image with the estimated parameters
            warped_image = dome.warp_image_for_dome([test_image])
            warped_image.save("warped_test_image_" + str(webcam_theta) + ".jpg",
                              "jpeg")
            




