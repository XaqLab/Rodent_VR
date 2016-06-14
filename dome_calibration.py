#!/usr/python

"""
Calculate the geometric parameters of our dome projection system from
a list of centroids belonging to spots in a projected image that match the
viewing directions of light produced by a 3D printed calibration device.

A calibration image is generated using caldev.exe such that the green spots in
the projector image are in the center of the white spots produced by a 3D
printed calibration device that emits light in known directions and is placed
at the animal's viewing position.  Since the center point of each object is
known, this provides the mapping from the set of center points to their
corresponding viewing directions.  The viewing direction for each center point
is also calculated using estimated parameter values and a minimization routine
is used to search for the true parameter values by minimizing the sum of the
lengths of the difference vectors between the actual viewing directions and
the estimated viewing directions.

To calculate the dome projection parameters, call this script with the file
saved from caldev.exe which contains a list of spot centroids.  Like this:

python dome_calibration.py centroid_list.txt 
    
"""

# import stuff from standard libraries
import sys
from PIL import Image
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan, arctan2
from numpy import array, uint8, dot, linalg, sqrt, float32, cross, zeros
from numpy.linalg import norm, inv
from scipy.optimize import minimize, fsolve
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_slsqp
from scipy.optimize import fmin_powell
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy import ndimage
import cv2

# import stuff for our projector setup and calibration camera
from dome_projection import NoViewingDirection
from dome_projection import DomeProjection
from dome_projection import flat_display_direction
from dome_projection import calc_projector_images
from dome_projection import calc_frustum_parameters
import foscam_FI9821P as camera

# define constants
MAX_FLOAT = sys.float_info.max
#PROJECTOR_PIXEL_WIDTH = 1280
#PROJECTOR_PIXEL_HEIGHT = 720
PROJECTOR_PIXEL_WIDTH = 1024
PROJECTOR_PIXEL_HEIGHT = 768
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


class InvalidParameters(Exception):
    def __init__(self):
         self.value = "Invalid Parameters!"
    def __str__(self):
        return repr(self.value)


def read_centroid_list(filename):
    """ Read the list of centroids for the light colored spots in the
    calibration image from a file saved using domecal.exe.  """
    try:
        centroid_file = open(filename, 'r')
        centroid_list = []
        for line in centroid_file:
            try:
                row, column = line.split(", ")
                centroid_list.append([float(row), float(column)])
            except:
                # ignore other lines in the file
                pass
        #print centroid_list
    except:
        print "Error reading list of centroids from", filename
        exit()
    return centroid_list


def print_parameters(parameters):
    """ print out the parameters that are being estimated """
    print "{'animal_y':%.3f," % parameters['animal_position'][1]
    print "'animal_z':%.3f," % parameters['animal_position'][2]
    print "'dome_y':%.3f," % parameters['dome_center'][1]
    print "'dome_z':%.3f," % parameters['dome_center'][2]
    print "'dome_radius':%.3f," % parameters['dome_radius']
    print "'mirror_radius':%.3f," % parameters['mirror_radius']
    print "'projector_y':%.3f," % parameters['projector_focal_point'][1]
    print "'projector_z':%.3f," % parameters['projector_focal_point'][2]
    print "'projector_roll':%.3f," % parameters['projector_roll']
    print "'projector_theta':%.3f," % parameters['projector_theta']
    print "'projector_vertical_offset':%.3f}" % parameters['projector_vertical_offset']


def parameters_to_x(parameters):
    """ convert a parameters dictionary to an array for minimization """
    x = (parameters['animal_position'][1],
         parameters['animal_position'][2],
         parameters['dome_center'][1],
         parameters['dome_center'][2],
         parameters['dome_radius'],
         parameters['mirror_radius'],
         parameters['projector_focal_point'][1],
         parameters['projector_focal_point'][2],
         parameters['projector_roll'],
         parameters['projector_theta'],
         parameters['projector_vertical_offset'])
    return x


def x_to_parameters(x):
    """ decode x into meaningful names and setup the parameters dictionary """
    animal_y = x[0]
    animal_z = x[1]
    dome_y = x[2]
    dome_z = x[3]
    dome_radius = x[4]
    mirror_radius = x[5]
    projector_y = x[6]
    projector_z = x[7]
    projector_roll = x[8]
    projector_theta = x[9]
    projector_vertical_offset = x[10]
    parameters = dict(image_pixel_width = [PROJECTOR_PIXEL_WIDTH],
                      image_pixel_height = [PROJECTOR_PIXEL_HEIGHT],
                      projector_pixel_width = PROJECTOR_PIXEL_WIDTH,
                      projector_pixel_height = PROJECTOR_PIXEL_HEIGHT,
                      projector_focal_point = [0, projector_y, projector_z],
                      projector_theta = projector_theta,
                      projector_vertical_offset = projector_vertical_offset,
                      projector_roll = projector_roll,
                      mirror_radius = mirror_radius,
                      dome_center = [0, dome_y, dome_z],
                      dome_radius = dome_radius,
                      animal_position = [0, animal_y, animal_z])
    return parameters
    

def dome_distortion(x, projector_points):
    """ Calculate the sum of the L2 norm of the difference vectors between the
    actual and estimated viewing directions.  """
    # debug stuff
    global previous_parameters
    global best_sum_of_errors
    global best_parameters
    """ convert x into the parameters dictionary """
    parameters = x_to_parameters(x)

    if previous_parameters:
        print
        print "Parameters"
        print_parameters(parameters)
        print
    previous_parameters = dict(parameters)

    try:
        """
        Find the actual viewing directions using photo_points and the estimated
        parameters.  Because the orientation of the camera is unknown,
        determination of the actual viewing directions depends on the estimates
        of the animal's position, the dome position and the dome radius.
        """
        dome = DomeProjection(**parameters)
        actual_directions = dome.calibration_directions
        """
        Calculate the viewing directions for projector_points using these
        parameter estimates.
        """
        estimated_directions = [zeros(3)]*len(actual_directions)
        for i in range(len(projector_points)):
            try:
                direction = dome.dome_display_direction(*projector_points[i])
                estimated_directions[i] = direction
            except NoViewingDirection:
                # For each point that has no viewing direction, set its
                # estimated viewing direction to the opposite of the actual
                # direction.  This will produce the worst possible result for
                # these points and encourage the minimization routine to look
                # elsewhere.
                estimated_directions[i] = -1*array(actual_directions[i])
        """
        Calculate the length of the difference between each measured direction
        and it's corresponding calculated direction.  Return the sum of these
        differences.
        """
        sum_of_errors = 0
        max_error = 0
        print "[actual pitch, actual yaw], [estimated pitch, estimated yaw],",
        print "estimated - actual"
        for i, actual_direction in enumerate(actual_directions):
            [x, y, z] = actual_direction
            yaw = 180/pi*arctan2(x, y)
            pitch = 180/pi*arcsin(z)
            print "[%6.3f, %6.3f]," % (pitch, yaw),
            error = linalg.norm(actual_direction - estimated_directions[i])
            sum_of_errors = sum_of_errors + error**2
            if error > max_error:
                max_error = error
            [x, y, z] = estimated_directions[i]
            est_yaw = 180/pi*arctan2(x, y)
            est_pitch = 180/pi*arcsin(z)
            print "[%6.3f, %6.3f]," % (est_pitch, est_yaw),
            print "[%6.3f, %6.3f]" % (est_pitch - pitch, est_yaw - yaw)
    
    except InvalidParameters:
        # Catch cases where the optimization routine tries some invalid
        # parameter values.  Usually this means the animal location is outside
        # the dome.  Return the max sum of errors to discourage this.
        sum_of_errors = 2*len(projector_points)

    print
    print "Sum of errors:", sum_of_errors
    print "Max error:", max_error
    if sum_of_errors < best_sum_of_errors:
        best_sum_of_errors = sum_of_errors
        best_parameters = dict(parameters)
    return sum_of_errors


def estimate_parameters(projector_points):
    """
    Search the parameter space to find values that minimize the sum of the
    length of the difference vectors between the measured directions, found
    using caldev.exe and the 3D printed calibration device, and the
    calculated directions.
    """
    # Setup initial values of the parameters
    dome = DomeProjection()
    parameters = dome.get_parameters()
    #from params_2015_11_04_17_16 import parameters # 1.7
    #from parameters_domecal import parameters # ??
    #from parameters_2015_02_10 import parameters # SLSQP: 1.7
    x0 = parameters_to_x(parameters)

    # Make sure mirror radius and dome radius are > 0, and limit the
    # animal's z-coordinate to keep it inside the dome
    x_bounds = [(-0.3, 0.3),   # animal position y-coordinate
                (0.3, 0.8),    # animal position z-coordinate
                (0.0, 0.3),    # dome center y-coordinate
                (0.2, 0.5),    # dome center z-coordinate
                (0.5, 0.7),    # dome radius
                (0.1, 0.3),    # mirror radius
                (0.5, 1.0),    # projector focal point y-coordinate
                (-0.2, 0.2),   # projector focal point z-coordinate
                (-0.1, 0.1),   # projector roll
                (0.1, 0.4),    # projector theta
                (0.0, 0.3)]    # projector vertical offset

    # Estimate parameter values by minimizing the difference between
    # the measured and calculated directions.
    arguments = tuple([projector_points])

    #results = basinhopping(dome_distortion, x0, niter=100, T=0.1,
                           #stepsize=0.005,
                           #minimizer_kwargs={'args':arguments},
                           #take_step=None, accept_test=None,
                           #callback=None, interval=50, disp=True,
                           #niter_success=None)

    #results = fmin_slsqp(dome_distortion, x0, args=arguments,
                         #bounds=x_bounds, acc=1e-6,
                         #full_output=True) # 0.003

    #results = minimize(dome_distortion, x0, args=arguments,
                       #method='TNC', bounds=x_bounds)
                       #method='Nelder-Mead') # 0.04
                       #method='L-BFGS-B', bounds=x_bounds)
    #results = fmin_l_bfgs_b(dome_distortion, x0, args=arguments, m=100,
                            #bounds=x_bounds, approx_grad=True, factr=1e0)

                            
    #import pdb; pdb.set_trace()
    results = fmin_l_bfgs_b(dome_distortion, x0, args=arguments,
                            bounds=x_bounds, approx_grad=True)
    print results

    # convert results into parameter dictionary
    try:
        # if results is a dictionary
        parameters = x_to_parameters(results['x'])
    except TypeError:
        # if results is a list
        parameters = x_to_parameters(results[0])
    return parameters


###############################################################################
# Main program starts here
###############################################################################
if __name__ == "__main__":
    """ Script was called from the command line so look for the arguments
    required to calculate the dome projection geometry """
    if len(sys.argv) != 2:
        """ Wrong number of arguments, display usage message """
        print "Usage:"
        print
        print sys.argv[0], "centroid_list.txt"
        print

    else:
        """ Calculate the dome projection parameters.  The first argument is
        the name of a file containing the list of centroids for the spots in
        the projected calibration image. """
        centroid_filename = sys.argv[1]
        """
        Read the list of centroids from the file specified by the first
        argument and convert them to (u,v) coordinates.  These (u,v) points are
        used to calculate viewing directions using the parameter estimates.
        """
        projector_centroids = read_centroid_list(centroid_filename)
        # convert (row, col) centroids to (u, v) points
        projector_points = [[c[1] + 0.5, c[0] + 0.5]
                            for c in projector_centroids]

        # search the parameter space to minimize differeces between measured
        # and calculated directions
        parameters = estimate_parameters(projector_points)

        # Print out the estimated parameter values
        print_parameters(parameters)


