#!/usr/python

"""
Calculate the dome projection parameters from calibration photos.  

A dome calibration image is projected onto the dome using domecal.exe and
four overlapping photographs are taken of it with a camera that has been
characterized for use in parameter estimation.  The calibration image contains
light colored objects, centered on known pixels, against a dark background.
The animal's viewing direction to each light colored object is calculated from
the photographs.  Since the center pixel of each object is known, this provides
the mapping from the set of center pixels to their corresponding viewing
directions.  The viewing direction for each center pixel is calculated using
estimated parameter values and a minimization routine is used to search for the
true parameter values by minimizing the sum of the square differences between
the actual viewing directions, calculated from the photos, and the estimated
viewing directions calculated using the estimated parameters.

To calculate the dome projection parameters, call this script with the file
saved from domecal.exe, which contains a list of the center pixels, and the
calibration photos as arguments.  Like this:

python dome_calibration.py pixel_list.txt pic1.jpg pic2.jpg pic3.jpg pic4.jpg
    
The images should be in this order: top left, top right, bottom left, bottom
right.
"""

# import stuff from standard libraries
import sys
from PIL import Image
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan, arctan2
from numpy import array, uint8, dot, linalg, sqrt, float32, cross
from numpy.linalg import norm, inv
from scipy.optimize import minimize, fsolve
from scipy import ndimage
import cv2

# import stuff for our projector setup and web camera
from dome_projection import DomeProjection
from dome_projection import flat_display_direction
from dome_projection import calc_projector_images
import foscam_FI9821P as camera

# define constants
MAX_FLOAT = sys.float_info.max
PROJECTOR_PIXEL_WIDTH = 1280
PROJECTOR_PIXEL_HEIGHT = 720
CAMERA_PIXEL_WIDTH = 1280
CAMERA_PIXEL_HEIGHT = 720
BACKGROUND_PIXEL_VALUE = 0
OBJECT_PIXEL_VALUE = 192  # < 255 to prevent saturating the camera
LOWER_LEFT_PHOTO = 0   # defined to make the code easier to read
LOWER_RIGHT_PHOTO = 1  # defined to make the code easier to read
UPPER_LEFT_PHOTO = 2   # defined to make the code easier to read
UPPER_RIGHT_PHOTO = 3  # defined to make the code easier to read

# debug stuff
DEBUG = True
previous_parameters = None
best_parameters = None
best_sum_of_errors = 100


def read_pixel_list(filename):
    """
    Read the list of center pixels for the light colored objects in the
    calibration image from a file saved using domecal.exe.
    """
    try:
        pixel_list_file = open(filename, 'r')
        pixel_list = []
        for line in pixel_list_file:
            try:
                row, column = line.split(", ")
                pixel_list.append([float(row), float(column)])
            except:
                # ignore lines with parameter values
                pass
        #print pixel_list
    except:
        print "Error reading pixel list from", filename
        exit()
    return pixel_list


def column_vector(vector):
    """
    take a list or array and make sure it's in column vector format for Open CV
    """
    shape = array(vector).shape
    assert len(shape) < 3, "Input vector must have less than three dimensions."
    if len(shape) > 1:
        assert 1 in shape
    length = len(vector)
    if shape == (length, 1):
        # input is a column vector
        column_vector = array(vector, dtype=float32)
    else:
        # input is a row vector
        column_vector = array([[i] for i in vector], dtype=float32)
    return column_vector


def pixels_to_points(pixels, pixel_width, pixel_height):
    """ Convert (row, column) pixels to (x, y) points. """
    points = [[pixels[i][1] - pixel_width/2 + 0.5,
               -(pixels[i][0] - pixel_height/2 + 0.5)]
              for i in range(len(pixels))]
    return points


def points_to_pixels(points, pixel_width, pixel_height):
    """ Convert (x, y) points to (row, column) pixels. """
    pixels = [[-point[1] + pixel_height/2 - 0.5,
                point[0] + pixel_width/2 - 0.5]
              for point in points]
    return pixels


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


def remove_distortion(points, coefficients):
    """
    Map distorted (x, y) points to undistorted (x, y) points using a simplified
    Brown-Conrady model.
    """
    DEBUG_DISTORTION_REMOVAL = False

    # calculate the undistorted x and y values from the distorted values
    x = array([points[i][0] for i in range(len(points))])
    y = array([points[i][1] for i in range(len(points))])
    K = coefficients
    r2 = x**2 + y**2 
    x_undistorted = x*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)
    y_undistorted = y*(1 + K[0]*r2 + K[1]*r2**2 + K[2]*r2**3)

    if DEBUG_DISTORTION_REMOVAL:
        print "\nPhoto center point distortion differences:"
        x_differeneces = x - x_undistorted
        y_differeneces = y - y_undistorted
        for i in range(len(points)):
            print "%3f, %3f" % (x_differeneces[i], y_differeneces[i]),
            if i % 6 == 5:
                print
        print

    return [[x_undistorted[i], y_undistorted[i]] for i in range(len(points))]


def find_center_pixels(image, blur=2, remove_distortion=True):
    """
    This function returns list of center pixels for light colored objects on a
    dark background in a grey scale camera photo.  The objects are
    distinguished from the background by using a threshold pixel value.
    """
    #image.show()
    pixels = array(image)
    
    # define the pixel threshold used to distinguish objects from the background
    # this value was determined empirically
    PIXEL_THRESHOLD = 0.4 * (int(pixels.max()) + int(pixels.min()))

    # smooth image to eliminate aliasing artifacts
    blurred_pixels = ndimage.gaussian_filter(pixels, sigma=blur)
    #Image.fromarray(blurred_pixels).show()

    # identify object pixels using the pixel threshold
    object_pixels = array(blurred_pixels > PIXEL_THRESHOLD, dtype=uint8)

    # label the objects in the image and find the center pixel for each object
    labeled_pixels, num_labels = ndimage.label(object_pixels)
    center_pixels = ndimage.center_of_mass(pixels, labeled_pixels,
                                           range(1, num_labels + 1))
    if False:
        """ invert the center pixels so they can be seen """
        for pixel in center_pixels:
            object_pixels[int(round(pixel[0])), int(round(pixel[1]))] = 0
        Image.fromarray(255*object_pixels).show()
    
    if remove_distortion:
        # Convert center_pixels to the format that Open CV requires
        opencv_pixels = array([[[p[1], p[0]]] for p in center_pixels],
                            dtype=float32)
        undist_pixels = cv2.undistortPoints(opencv_pixels, camera.matrix,
                                            camera.distortion_coefficients,
                                            R=None, P=camera.matrix)
        # Convert undist_pixels back to a simple array of pixels
        center_pixels = array([[p[0, 1], p[0, 0]] for p in undist_pixels])

        if False:
            """ invert the center pixels so they can be seen """
            for pixel in center_pixels:
                object_pixels[int(round(pixel[0])), int(round(pixel[1]))] = \
                        0x80
            Image.fromarray(255*object_pixels).show()

    return center_pixels


def rotate(vector, rotation_vector):
    """ rotate vector around rotation_vector by an angle equal to the magnitude
    of rotation_vector (clockwise rotation when looking in the rotation_vector
    direction) """
    rotation_matrix = cv2.Rodrigues(column_vector(rotation_vector))[0]
    return rotation_matrix.dot(vector)


def calc_distance_to_dome(parameters, position, direction):
    """
    Find the distance from a position inside the dome to the dome in the given
    direction.  This boils down to completing a triangle, in which the length
    of one side is unknown, using the law of cosines. This gives a quadratic
    equation for the unknown length and this equation is solved and the
    positive root is taken.  If both roots are imaginary the line does not
    intersect the dome.  This should be prevented by judicious choice of 
    parameter bounds.
    """
    assert (linalg.norm(direction) - 1.0) < 1e-6, linalg.norm(direction) - 1.0
    dome_center = parameters['dome_center']
    dome_radius = parameters['dome_radius']
    position_to_dome_center = array(dome_center) - array(position)
    distance_to_dome_center = linalg.norm(position_to_dome_center)
    if distance_to_dome_center > 0:
        theta = arccos(position_to_dome_center.dot(direction) /
                       distance_to_dome_center)
    else:
        # avoid division by zero
        theta = 0
    a = 1.0
    b = -2*linalg.norm(position_to_dome_center)*cos(theta)
    c = linalg.norm(position_to_dome_center)**2 - dome_radius**2
    if b**2 - 4*a*c < 0:
        # no solution, start debugger to see what's going on
        import pdb; pdb.set_trace()
    d = sqrt(b**2 - 4*a*c)
    r = max([(-b + d) / (2*a), (-b - d) / (2*a)])
    return r


def find_viewing_direction(photo_pixel, camera_direction, parameters):
    """
    Use the camera orientation and the direction to a pixel in the photo
    to find the (x, y, z) coordinates of the pixel on the dome.
    """
    # find the camera's coordinate system in terms of the dome coordinate
    # system
    ccz = camera_direction
    ccx_vector = cross([0, 0, -1], ccz)
    ccx = ccx_vector/norm(ccx_vector)
    ccy = cross(ccz, ccx)
    # find the camera's reference vector in dome coordinates
    reference_vector = (camera.reference_vector[0]*ccx +
                        camera.reference_vector[1]*ccy +
                        camera.reference_vector[2]*ccz)
    animal_position = parameters['animal_position']
    camera_focal_point = animal_position - reference_vector
    [x, y, z] = camera_direction
    camera_pitch = arcsin(z)
    camera_yaw = arctan2(x, y)
    photo_point = array([photo_pixel[1], photo_pixel[0], 1])
    # find the vector to this point in camera coordinates
    point_vector = inv(camera.matrix).dot(photo_point)
    # rotate the vector to match the dome coordinate system and account for
    # camera orientation
    point_vector = rotate(point_vector, array([camera_pitch - pi/2, 0, 0]))
    point_vector = rotate(point_vector, array([0, 0, -camera_yaw]))
    #rotation_vector = array([-pi/2, 0, 0])
    #rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    #point_vector = rotation_matrix.dot(point_vector)
    point_direction = point_vector / norm(point_vector)
    [x, y, z] = point_direction
    point_pitch = arcsin(z)
    point_yaw = arctan2(x, y)
    #print "Point pitch:", 180/pi*point_pitch
    #print "Point yaw:", 180/pi*point_yaw
    distance_to_dome = calc_distance_to_dome(parameters,
                                             camera_focal_point,
                                             point_direction)
    vector_to_dome = distance_to_dome * point_direction
    dome_point = camera_focal_point + vector_to_dome
    """
    Find the viewing direction from the animal's (x, y, z) position to the
    (x, y, z) point on the dome.
    """
    viewing_vector = dome_point - animal_position
    viewing_direction = (viewing_vector / linalg.norm(viewing_vector))
    return viewing_direction


def camera_orientation_error(orientation, lower_left_pixel, known_direction,
                             parameters):
    """
    Calculate the pitch and yaw of lower_left_pixel and compare it to the
    pitch and yaw of known_direction.  The camera orientation is correct when
    they are the same.
    """
    camera_pitch, camera_yaw = orientation
    camera_direction = array([cos(camera_pitch)*sin(camera_yaw),
                              cos(camera_pitch)*cos(camera_yaw),
                              sin(camera_pitch)])
    viewing_direction = find_viewing_direction(lower_left_pixel,
                                               camera_direction,
                                               parameters)
    """
    viewing direction and known direction should be equal
    """
    return norm(viewing_direction - known_direction)
    

def calc_viewing_directions(photo_pixels, parameters):
    """
    Calculate the viewing directions to the light colored objects in the
    calibration photos.  
    """
    [upper_left, upper_right, lower_left, lower_right] = photo_pixels
    middle_row = camera.pixel_height/2 + 0.5
    middle_col = camera.pixel_width/2 + 0.5
    """
    Start with the lower left photo because it is the only one that contains
    an object with a known viewing direction.  The lower left object in this
    photo is at 0 pitch, 0 yaw.
    """
    photos = [lower_left, lower_right, upper_left, upper_right]
    known_directions = [[0.0, 1.0, 0.0]] * 4
    viewing_directions = [[0.0, 1.0, 0.0]] * 9
    animal_position = parameters['animal_position']
    for photo in range(len(photos)):
        #print "Photo number:", photo
        for pixel in photos[photo]:
            [row, col] = pixel
            if row > middle_row and col < middle_col:
                lower_left_pixel = pixel
                #print "Lower left pixel:", lower_left_pixel
        """
        Find the camera orientation for this photo.
        """
        [x, y, z] = known_directions[photo]
        known_pitch = arcsin(z)
        known_yaw = arctan2(x, y)
        #print "Known pitch, yaw:    %11f, %11f" % (180/pi*known_pitch,
                                                   #180/pi*known_yaw)
        apparent_direction = find_viewing_direction(lower_left_pixel,
                                                    array([0, 1, 0]),
                                                    parameters)
        [x, y, z] = apparent_direction
        apparent_pitch = arcsin(z)
        apparent_yaw = arctan2(x, y)
        #print "Apparent pitch, yaw: %11f, %11f"  % (180/pi*apparent_pitch,
                                                    #180/pi*apparent_yaw)
        starting_pitch = known_pitch - apparent_pitch
        starting_yaw = known_yaw - apparent_yaw
        #print "Starting pitch, yaw: %11f, %11f" % (180/pi*starting_pitch,
                                                   #180/pi*starting_yaw)
        starting_orientation = (starting_pitch, starting_yaw)
        args = (lower_left_pixel, known_directions[photo], parameters)
        results = minimize(camera_orientation_error, starting_orientation,
                           args=args, method='Nelder-Mead',
                           options={'xtol':1e-9})
        camera_pitch, camera_yaw = results['x']
        #print "Camera pitch, yaw:   %11f, %11f" % (180/pi*camera_pitch,
                                                   #180/pi*camera_yaw)
        #import pdb; pdb.set_trace()
        camera_direction = array([cos(camera_pitch)*sin(camera_yaw),
                                  cos(camera_pitch)*cos(camera_yaw),
                                  sin(camera_pitch)])
        for pixel in photos[photo]:
            """
            Find viewing directions from the animal's (x, y, z) position to the
            (x, y, z) locations of the object's center pixels on the dome.
            """
            viewing_direction = find_viewing_direction(pixel,
                                                       camera_direction,
                                                       parameters)
            """
            The lower left photo provides the known directions for the
            other three photos.  The viewing_directions are indexed, 0 through
            8 from left to right and then bottom to top.  This is necessary to
            match the order of projector pixels in the calibration image.
            """
            [row, col] = pixel
            if row > middle_row and col > middle_col:
                # lower right pixel
                if photo == LOWER_LEFT_PHOTO:
                    viewing_directions[1] = viewing_direction
                    # known direction for the lower right photo
                    known_directions[1] = viewing_direction
                elif photo == LOWER_RIGHT_PHOTO:
                    viewing_directions[2] = viewing_direction
            elif row < middle_row and col < middle_col:
                # upper left pixel
                if photo == LOWER_LEFT_PHOTO:
                    viewing_directions[3] = viewing_direction
                    # known direction for the upper left photo
                    known_directions[2] = viewing_direction
                elif photo == UPPER_LEFT_PHOTO:
                    viewing_directions[6] = viewing_direction
            elif row < middle_row and col > middle_col:
                # upper right pixel
                if photo == LOWER_LEFT_PHOTO:
                    viewing_directions[4] = viewing_direction
                    # known direction for the upper right photo
                    known_directions[3] = viewing_direction
                elif photo == LOWER_RIGHT_PHOTO:
                    viewing_directions[5] = viewing_direction
                elif photo == UPPER_LEFT_PHOTO:
                    viewing_directions[7] = viewing_direction
                elif photo == UPPER_RIGHT_PHOTO:
                    viewing_directions[8] = viewing_direction
    # re-arrange the order of the viewing directions to match the pixel
    # list, left to right, top to bottom
    top_row = viewing_directions[6:9]
    middle_row = viewing_directions[3:6]
    bottom_row = viewing_directions[0:3]
    viewing_directions = top_row + middle_row + bottom_row
    return viewing_directions


def calc_frustum_parameters(image1, image2):
    """
    Inverse of calc_projector_images.
    This funciton calculates the y and z coordinates of the projector's focal
    point along with it's horizontal field of view, and it's vertical throw.
    This is done to reduce the degrees of freedom to the minimum necessary for
    parameter estimation using SciPy's minimization routine.
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


def dome_distortion(x, projector_pixels, photo_pixels):
    # debug stuff
    global previous_parameters
    global best_sum_of_errors
    global best_parameters
    """
    Calculate the sum of the square differences between the actual and
    estimated viewing directions.
    """
    # there are 9 spots in the calibration image
    assert len(projector_pixels) == 9
    # 4 overlapping photos are taken, each containing 4 spots
    assert len(photo_pixels) == 4
    """ decode x into meaningful names and setup the parameters dictionary """
    projector_y = x[0]
    projector_z = x[1]
    projector_theta = x[2]
    vertical_offset = x[3]
    projector_images = calc_projector_images(projector_y, projector_z,
                                             projector_theta, vertical_offset)
    projector_roll = x[4]
    mirror_radius = x[5]
    dome_y = x[6]
    dome_z = x[7]
    dome_radius = x[8]
    animal_y = x[9]
    animal_z = x[10]
    parameters = dict(image_pixel_width = [1280],
                      image_pixel_height = [720],
                      projector_pixel_width = 1280,
                      projector_pixel_height = 720,
                      first_projector_image = projector_images[0],
                      second_projector_image = projector_images[1],
                      projector_roll = projector_roll,
                      mirror_radius = mirror_radius,
                      dome_center = [0, dome_y, dome_z],
                      dome_radius = dome_radius,
                      animal_position = [0, animal_y, animal_z])

    if previous_parameters:
        print
        print "Parameters that changed"
        for parameter in parameters:
            if parameters[parameter] != previous_parameters[parameter]:
                print parameter
                print previous_parameters[parameter]
                print parameters[parameter]
        print
    previous_parameters = dict(parameters)

    """
    Find the actual viewing directions using photo_pixels and the estimated
    parameters.  Because the orientation of the camera is unknown,
    determination of the actual viewing directions depends on the estimates
    of the animal's position, the dome position and the dome radius.
    """
    actual_directions = calc_viewing_directions(photo_pixels, parameters)
    """
    Calculate the viewing directions for projector_pixels using these
    parameter estimates.
    """
    dome = DomeProjection(**parameters)
    estimated_directions = [dome.dome_display_direction(pixel[0], pixel[1])[1]
                            for pixel in projector_pixels]
    """
    Calculate the length of the difference between each measured direction and
    it's corresponding calculated direction.  Return the sum of these
    differences.
    """
    sum_of_errors = 0
    print "[actual pitch, actual yaw], [estimated pitch, estimated yaw]"
    for i, actual_direction in enumerate(actual_directions):
        [x, y, z] = actual_direction
        yaw = 180/pi*arctan2(x, y)
        pitch = 180/pi*arcsin(z)
        print "[%f, %f]," % (pitch, yaw),
        error = linalg.norm(actual_direction - estimated_directions[i])
        sum_of_errors = sum_of_errors + error
        [x, y, z] = estimated_directions[i]
        yaw = 180/pi*arctan2(x, y)
        pitch = 180/pi*arcsin(z)
        print "[%f, %f]" % (pitch, yaw)

    print
    print "Sum of errors:", sum_of_errors
    if sum_of_errors < best_sum_of_errors:
        best_sum_of_errors = sum_of_errors
        best_parameters = dict(parameters)
    #import pdb; pdb.set_trace()
    return sum_of_errors


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":

    if len(sys.argv) != 6:
        """
        Wrong number of arguments, display usage message
        """
        print "Usage:"
        print
        print sys.argv[0], "pixel_list.txt pic1 pic2 pic3 pic4"
        print
        print "the picture order is: upper left, upper right,",
        print "lower left, lower right"

    else:
        """
        Calculate the dome projection parameters.  The first argument is the
        name of a file containing the list of projector pixels upon which the
        objects in the calibration image are centered.  The second through
        fifth arguments are the file names of the calibration photos.
        """
        pixel_list_filename = sys.argv[1]
        photo_filenames = sys.argv[2:]
        """
        Read the list of projector pixels from the file specified by the first
        argument.  These pixels are used to calculate viewing directions using
        the parameter estimates.
        """
        projector_pixels = read_pixel_list(pixel_list_filename)
        """
        Find the center points of the light colored objects in the calibration
        photos and compensate for the radial distortion of the camera.  These
        points are used to find the actual viewing directions for the projector
        pixels in projector_pixels.
        """
        # Find the center points of the objects in the calibration photos.
        photo_pixels = []
        for photo_filename in photo_filenames:
            photo = Image.open(photo_filename).convert('L')
            pixels = find_center_pixels(photo)
            photo.close()
            photo_pixels.append(pixels)

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
        projector_roll = 0
        mirror_radius = 0.215
        dome_center = [0, 0.138, 0.309]
        dome_radius = 0.603
        animal_position = [0, 0.06, 0.61]
        
        x0 = (calc_frustum_parameters(first_projector_image,
                                      second_projector_image) + 
              [projector_roll,
               mirror_radius,
               dome_center[1],
               dome_center[2],
               dome_radius,
               animal_position[1],
               animal_position[2]])

        # Make sure mirror radius and dome radii are > 0, and limit the
        # animal's z-coordinate to keep it inside the dome
        parameter_bounds = [(None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None),
                            (0, None),
                            (None, None),
                            (None, None),
                            (0, None),
                            (None, None),
                            (None, 0.6)]

        # Estimate parameter values by minimizing the difference between
        # the measured and calculated directions.
        arguments = (projector_pixels, photo_pixels)
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
        projector_roll = results['x'][4]
        mirror_radius = results['x'][5]
        dome_y = results['x'][6]
        dome_z = results['x'][7]
        dome_radius = results['x'][8]
        animal_y = results['x'][9]
        animal_z = results['x'][10]

        # Print out the estimated parameter values
        for image in projector_images:
            for row in image:
                print row
        print "Projector roll:", projector_roll
        print "Mirror radius:", mirror_radius
        print "Dome y-coordinate:", dome_y
        print "Dome z-coordinate:", dome_z
        print "Dome radius:", dome_radius
        print "Animal y-coordinate:", animal_y
        print "Animal z-coordinate:", animal_z


        exit() 


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
        


