#!/usr/python

"""
Estimate the dome projection parameters.  

The dome calibration image is produced when this script is run with no
arguments.  This image is projected onto the dome and photographed with a camera
for parameter estimation.

When the calibration image and the photograph of the calibration image are passed
to this script as arguments, it returns the estimated parameters.
"""

# import stuff from standard libraries
import sys
from PIL import Image
from numpy import array, ones, uint8, mean, dot, tan, arctan, sin, cos, linalg
from scipy.optimize import minimize
from scipy import ndimage

# import stuff for our projector setup and web camera
from dome_projection import DomeProjection
from dome_projection import flat_display_direction
import webcam


DEBUG = True

# define constants
PROJECTOR_PIXEL_WIDTH = 1280
PROJECTOR_PIXEL_HEIGHT = 720
CAMERA_PIXEL_WIDTH = 1280
CAMERA_PIXEL_HEIGHT = 720
BACKGROUND_PIXEL_VALUE = 0
OBJECT_PIXEL_VALUE = 192  # < 255 to prevent saturating the camera


def create_diamond_image(center_pixels, diamond_size,
                         image_width=PROJECTOR_PIXEL_WIDTH,
                         image_height=PROJECTOR_PIXEL_HEIGHT):
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


def create_calibration_image():
    """
    Make a calibration image which projects spots of light on the dome at
    desired animal viewing angles, described by yaw and pitch.  These viewing
    angles depend on the defaulte parameter values in dome_projection.
    """
    # find the pixels that go with those directions
    dome = DomeProjection()
    # guess some pixel values
    #pixels = [[469, 1160], [547, 948], [569, 640], [547, 331], [469, 119],
    #          [300, 895], [360, 804], [380, 639], [360, 475], [300, 384]]
    pixels = [[463, 1163], [543, 943], [563, 643], [543, 333], [463, 113],
              [303, 893], [363, 803], [383, 633], [363, 473], [303, 383]]
    calibration_pixels = \
            dome.find_projector_pixels(dome.calibration_directions, pixels)
    print calibration_pixels

    # create the calibration image using this center pixels
    calibration_image = create_diamond_image(calibration_pixels, 1)
    calibration_image.show()

    return calibration_image


def sort_points(points, num_rows, num_columns):
    """
    Take a set of points that are known to be arranged in a grid layout and
    that are already sorted by their y values and sort them from left to right
    and top to bottom.
    """
    rows = []
    for r in range(num_rows):
        rows.append(points[r * num_columns : (r + 1) * num_columns])
        # sort each row of points by x value
        rows[-1].sort(key=lambda p:p[0])

    sorted_points = []
    for row in rows:
        sorted_points.extend(row)

    return sorted_points


def find_centers(image, blur=2):
    """
    This function returns coordinates for the centers of objects in a grey
    scale image.  The objects are distinguished from the background by using
    a threshold pixel value.  The coordinates are ordered from left to
    right and then top to bottom.  The coordinates can be returned in either
    the default (x, y) format or in (row, column) format.
    """

    image.show()
    [image_width, image_height] = image.size
    pixels = array(image)
    
    # define the pixel threshold used to distinguish objects from the background
    # this value was determined empirically
    PIXEL_THRESHOLD = 0.4 * (int(pixels.max()) + int(pixels.min()))

    # smooth image to eliminate aliasing artifacts
    blurred_pixels = ndimage.gaussian_filter(pixels, sigma=blur)
    Image.fromarray(blurred_pixels).show()

    # identify object pixels using the pixel threshold
    object_pixels = array(blurred_pixels > PIXEL_THRESHOLD, dtype=uint8)

    # label the objects in the image and find the center pixel for each object
    labeled_pixels, num_labels = ndimage.label(object_pixels)
    center_pixels = ndimage.center_of_mass(pixels, labeled_pixels,
                                           range(1, num_labels + 1))

    if DEBUG:
        """ invert the center pixels so they can be seen """
        for pixel in center_pixels:
            object_pixels[int(round(pixel[0])), int(round(pixel[1]))] = 0
        Image.fromarray(255*object_pixels).show()
    
    # convert center pixels (row, column) to center points (x, y)
    center_points = []
    for pixel in center_pixels:
        x_center = pixel[1] - image_width/2 + 0.5
        y_center = -(pixel[0] - image_height/2 + 0.5)
        center_points.append([x_center, y_center])

    # need to sort the points from left to right and then top to bottom so they
    # can be matched up with their equivalent points
    return center_points


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


def dome_distortion(x, image_pixels, photo_pixels):
    """
    Calculate the sum of the square differences between measured and calculated
    directions.
    """
    assert len(image_pixels) == len(photo_pixels)
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

    # Calculate the viewing directions to photo_pixels using the camera's
    # estimated field of view.
    measured_directions = []
    for photo_pixel in photo_pixels:
        [row, column] = photo_pixel
        arguments = [row, column, webcam.screen_height, webcam.screen_width,
                     webcam.pixel_height, webcam.pixel_width,
                     webcam.distance_to_screen]
        measured_directions.append(flat_display_direction(*arguments))

    # setup the parameters dictionary
    parameters = dict(image_pixel_width = [1280],
                      image_pixel_height = [720],
                      projector_pixel_width = 1280,
                      projector_pixel_height = 720,
                      first_projector_image = projector_images[0],
                      second_projector_image = projector_images[1],
                      mirror_radius = mirror_radius,
                      dome_center = [0, dome_y, dome_z],
                      dome_radius = dome_radius,
                      animal_position = [0, animal_y, animal_z])

    # Calculate the viewing directions for image_pixels using these parameters.
    dome = DomeProjection(**parameters)
    calculated_directions = []
    for image_pixel in image_pixels:
        [row, col] = image_pixel
        calculated_direction = dome.dome_display_direction(row, col)[1]
        calculated_directions.append(calculated_direction)
    #if DEBUG:
    if False:
        print "Measured directions:", measured_directions
        print "Calculated directions:", calculated_directions

    # for each measured direction, find the closest calculated direction
    # to calculate errors
    sum_of_square_errors = 0
    for measured_direction in measured_directions:
        # find minimum error (i.e. closest photo point)
        errors = [linalg.norm(measured_direction - calculated_direction)
                  for calculated_direction in calculated_directions]
        min_error = min(errors)
        # add minimum error to the running total
        sum_of_square_errors = sum_of_square_errors + min_error
        # remove that calculated direction
        index = errors.index(min_error)
        calculated_directions.pop(index)

    # value = sum([linalg.norm(measured_directions[i] - calculated_directions[i])
    #                          for i in range(len(measured_directions))])
    # return value
    return sum_of_square_errors


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":

    if len(sys.argv) == 1:
        """
        No arguments were given so generate the calibration image and save
        it to a file.  The calibration image is projected on the dome and
        photographed with a camera in order to estimate the dome projection
        parameters.
        """
        print "Creating calibration image..."
        dome_image = create_calibration_image()
        dome_image.save("dome_calibration_image.png")

    elif len(sys.argv) == 3:
        """
        Two arguments were given so estimate the dome projection parameters.
        Treat the first argument as the file name of the calibration image
        and the second as the file name of the photograph of this image
        projected onto the dome.
        """
        image_filename = sys.argv[1]
        photo_filename = sys.argv[2]

        """
        Find the center pixels of the objects in the dome calibration photo
        and remove distortion introduced by the camera.  Then calculate the
        directions from the camera's focal point to these pixels and search
        the parameter space for parameter values that produce calculated
        directions for these pixels that match the measured directions.
        """
        # Find the center pixels of the objects in the calibration image.
        image = Image.open(image_filename).convert('L')
        image_centers = find_centers(image)
        image.close()

        # Convert (x, y) center points to (row, column) center pixels
        image_pixels = [[-center_point[1] + PROJECTOR_PIXEL_HEIGHT/2 - 0.5,
                         center_point[0] + PROJECTOR_PIXEL_WIDTH/2 - 0.5]
                        for center_point in image_centers]

        # Find the center pixels of the objects in the photograph of the
        # calibration image projected onto the dome.
        photo = Image.open(photo_filename).convert('L')
        photo_centers = find_centers(photo)
        photo.close()

        if DEBUG:
            distorted_centers = photo_centers

        # Remove the camera's distortion
        photo_centers = remove_distortion(photo_centers,
                                          webcam.distortion_coefficients)

        # Convert (x, y) center points to (row, column) center pixels
        photo_pixels = [[-center_point[1] + CAMERA_PIXEL_HEIGHT/2 - 0.5,
                         center_point[0] + CAMERA_PIXEL_WIDTH/2 - 0.5]
                        for center_point in photo_centers]

        if False:
            print "Dome image pixels:"
            print image_pixels
            print "\nDome calibration photo center point distortion differences:"
            for i in range(len(photo_centers)):
                x_differenece = distorted_centers[i][0] - photo_centers[i][0]
                y_differenece = distorted_centers[i][1] - photo_centers[i][1]
                print "%3f, %3f" % (x_differenece, y_differenece),
                if i % 6 == 5:
                    print
            print
            print "\nDome photo center pixels:"
            print photo_pixels

        """
        Search the parameter space to find values that minimize the square
        differences between the measured directions and calculated directions.
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

        # Estimate parameter values by minimizing the difference between
        # the measured and calculated directions.
        arguments = (image_pixels, photo_pixels)
        results = minimize(dome_distortion, x0, args=arguments,
                           method='L-BFGS-B', bounds=parameter_bounds)
        print results

        # Sort results into meaningful parameter names
        projector_y = results['x'][0]
        projector_z = results['x'][1]
        projector_theta = results['x'][2]
        vertical_offset = results['x'][3]
        projector_images = calc_projector_images(projector_y,
                                                 projector_z,
                                                 projector_theta,
                                                 vertical_offset)
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

        # Instantiate DomeProjection class with the estimated parameters and
        # warp a test image to see how it looks.
        #test_image = Image.open("test_images/WebCameras/Image42.jpg")
        test_image = Image.open("test_images/512_by_512/vertical_lines_16.png")

                              #screen_height=[screen_height],
                              #screen_width=[screen_width],
                              #distance_to_screen=[distance_to_screen],
        dome = DomeProjection(
                              screen_height = [1, 1, 1],
                              screen_width = [1, 1, 1],
                              distance_to_screen = [0.5, 0.5, 0.5],
                              pitch = [0, 0, 0],
                              yaw = [-90, 0, 90],
                              roll = [0, 0, 0],
                              image_pixel_width = [512, 512, 512],
                              image_pixel_height = [512, 512, 512],
                              first_projector_image=projector_images[0],
                              second_projector_image=projector_images[1],
                              mirror_radius=mirror_radius,
                              dome_center=[0, dome_y, dome_z],
                              animal_position = [0, animal_y, animal_z])

        # Warp a test image with the estimated parameters
        warped_image = dome.warp_image_for_dome([test_image, test_image,
                                                 test_image])
        warped_image.save("warped_test_image.jpg", "jpeg")
        


