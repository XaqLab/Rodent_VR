#!/usr/python

"""
Calculate the geometric parameters of our dome projection system from
a list of centroids belonging to spots in a projected image that match the
viewing directions of light produced by a 3D printed calibration device.

A calibration image is generated using caldev.exe such that the green dots in
the projector image are in the center of the white spots produced by a 3D
printed calibration device that emits light in known directions and is placed
at the animal's viewing position.  Since the center point of each green dot is
known, this provides the mapping from the set of center points to their
corresponding viewing directions.  The viewing direction for each center point
is also calculated using estimated parameter values and a minimization routine
is used to search for the true parameter values. Vectors representing the
difference between the actual viewing directions and the estimated viewing
directions are computed and the lengths of these vectors are squared and then
summed. This minimization routine estimates the geometric parameters of the
system by minimizing this sum.

To calculate the dome projection parameters, call this script with the file
saved from caldev.exe which contains a list of spot centroids.  Like this:

python dome_calibration.py centroid_list.txt 
    
"""

# import stuff from standard libraries
import sys
from datetime import datetime
from numpy import pi, sin, cos, arcsin, arctan2
from numpy import array, uint8, zeros
from numpy.linalg import norm
from PIL import Image
import cPickle as pickle
from scipy.optimize import fmin_l_bfgs_b
#from scipy.optimize import minimize, fsolve
#from scipy.optimize import fmin_slsqp
#from scipy.optimize import fmin_powell
#from scipy.optimize import basinhopping

# import stuff for our projector setup
from dome_projection import NoViewingDirection
from dome_projection import DomeProjection

# debug stuff
DEBUG = True
previous_parameters = None
best_parameters = None
best_sum_of_errors = 100


def get_projector_resolution():
    print "Please choose the projector's resolution from this list:"
    print "1) 1920 by 1080"
    print "2) 1280 by 720"
    print "3) 1024 by 768"
    done = False
    while not done:
        choice = raw_input("Choice: ")
        if "1" in choice:
            PROJECTOR_PIXEL_WIDTH = 1920
            PROJECTOR_PIXEL_HEIGHT = 1080
            done = True
        elif "2" in choice:
            PROJECTOR_PIXEL_WIDTH = 1280
            PROJECTOR_PIXEL_HEIGHT = 720
            done = True
        elif "3" in choice:
            PROJECTOR_PIXEL_WIDTH = 1024
            PROJECTOR_PIXEL_HEIGHT = 768
            done = True
    return PROJECTOR_PIXEL_WIDTH, PROJECTOR_PIXEL_HEIGHT


def compute_directions(pitch_angles, yaw_angles):
    """ Returns the directions used for calibration.
    For our 3D printed calibration device pitch_angles is [60, 30, 0, -15] and
    yaw_angles is [-120, -90, -60, -30, 0, 30, 60, 90, 120] or they are
    subsets of these. """
    calibration_directions = []
    # add straight up
    calibration_directions.append([0, 0, 1])
    for pitch in pitch_angles:
        for yaw in yaw_angles:
            x = sin(yaw * pi/180) * cos(pitch * pi/180)
            y = cos(yaw * pi/180) * cos(pitch * pi/180)
            z = sin(pitch * pi/180)
            calibration_directions.append([x, y, z])
    return calibration_directions


def check_point_symmetry(points):
    """ Check the symmetry of the points in the projector image to be used
    for calibration.  This will give us an idea of how well aligned the
    projection system is.  """
    # get the calibration directions from DomeProjection
    dome = DomeProjection()
    if len(points) == 22:  # 3 rows of 7 plus one over head
        pitch_angles = [60, 30, 0]
        yaw_angles = [-90, -60, -30, 0, 30, 60, 90]
    elif len(points) == 37:  # 4 rows of 9 plus one over head
        pitch_angles = [60, 30, 0, -15]
        yaw_angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    calibration_directions = compute_directions(pitch_angles, yaw_angles)
    
    # compare zero yaw points
    print "Zero yaw points:"
    print "(%6s, %6s)   (%6s, %6s)" % ("x", "y", "pitch", "yaw")
    for i in range(len(calibration_directions)):
        direction = calibration_directions[i]
        x,y,z = direction
        pitch = 180/pi*arcsin(z)
        yaw = 180/pi*arctan2(x, y)
        if abs(yaw - 0.0) < 15.0:
            point = points[i]
            x, y = point
            # shift row and column numbers so (0,0) is at the image center
            shifted_x = x - PROJECTOR_PIXEL_WIDTH/2
            shifted_y = y - PROJECTOR_PIXEL_HEIGHT/2
            print "(%6.1f, %6.1f)   (%6.1f, %6.1f)" % (shifted_x, shifted_y,
                                                       pitch, yaw)
    print
    
    # Compare symmetric points, e.g. (30, -60) and (30, 60) pitch and yaw.
    # If the dome projection system is aligned then the shifted y values should
    # be the same and the shifted x values should be opposites (one is the
    # negative of the other).
    print "Symmetric points (equal pitch, opposite yaw):"
    for i in range(len(calibration_directions)):
        direction1 = calibration_directions[i]
        x,y,z = direction1
        pitch1 = 180/pi*arcsin(z)
        yaw1 = 180/pi*arctan2(x, y)
        if yaw1 - 0.0 > 15.0:
            # positive non-zero yaw (negative will be the symmetric point)
            for j in range(len(calibration_directions)):
                direction2 = calibration_directions[j]
                x,y,z = direction2
                pitch2 = 180/pi*arcsin(z)
                yaw2 = 180/pi*arctan2(x, y)
                if abs(pitch1 - pitch2) < 15.0 and abs(yaw1 + yaw2) < 15.0:
                    # found symmetric points
                    point1 = points[i]
                    x1, y1 = point1
                    point2 = points[j]
                    x2, y2 = point2
                    # shift x and y values so (0,0) is at the image center
                    shifted_x1 = x1 - PROJECTOR_PIXEL_WIDTH/2
                    shifted_y1 = y1 - PROJECTOR_PIXEL_HEIGHT/2
                    shifted_x2 = x2 - PROJECTOR_PIXEL_WIDTH/2
                    shifted_y2 = y2 - PROJECTOR_PIXEL_HEIGHT/2
                    print "(%6.1f, %6.1f)   (%6.1f, %6.1f)" % \
                            (shifted_x1, shifted_y1, pitch1, yaw1)
                    print "(%6.1f, %6.1f)   (%6.1f, %6.1f)" % \
                            (shifted_x2, shifted_y2, pitch2, yaw2)
                    print
    

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
        The actual viewing directions corresponding to projector_points are
        contained in DomeProjection's calibration_directions property.  These
        directions are determined by the 3D printed calibration device.
        """
        dome = DomeProjection(**parameters)
        if len(projector_points) == 22:  # 3 rows of 7 plus one over head
            pitch_angles = [60, 30, 0]
            yaw_angles = [-90, -60, -30, 0, 30, 60, 90]
        elif len(projector_points) == 37:  # 4 rows of 9 plus one over head
            pitch_angles = [60, 30, 0, -15]
            yaw_angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
        actual_directions = compute_directions(pitch_angles, yaw_angles)
    
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
            error = norm(actual_direction - estimated_directions[i])
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


def save_calibration_image(parameters, datetime_string):
    """ Generate the projector calibration image to use with the 3D printed
    calibration device """
    filename = "calibration_image_" + datetime_string + ".png"
    print
    print "Saving calibration image in " + filename
    GREEN = array([0, 255, 0], dtype=uint8)
    
    dome = DomeProjection(**parameters)
    pitch_angles = [60, 30, 0, -15]
    yaw_angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    calibration_directions = compute_directions(pitch_angles, yaw_angles)
    pickle_filename = 'calibration_image.pkl'
    try:
        # see if there are previously found centroids that we can use as an initial
        # guess
        with open(pickle_filename, 'rb') as pickle_file:
            centroids = pickle.load(pickle_file)
        if len(centroids) == len(calibration_directions):
            centroids = dome.find_projector_points(calibration_directions,
                                                   centroids)
        else:
            # wrong number of previous centroids so do the search from scratch
            centroids = dome.find_projector_points(calibration_directions)
    except IOError:
        # no previous centroids found so do the search from scratch
        centroids = dome.find_projector_points(calibration_directions)
    
    # save centroids to a file for use as the initial guess next time
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(centroids, pickle_file)
    
    # make an image with 4 pixel squares centered on these pixel coordinates
    pixels = zeros([PROJECTOR_PIXEL_HEIGHT, PROJECTOR_PIXEL_WIDTH, 3], dtype=uint8)
    
    for centroid in centroids:
        # convert centroids from (u, v) coordinates to (row, col) coordinates
        col, row = centroid
        row = int(round(row - 0.5))
        col = int(round(col - 0.5))
        if (row >= 0 and row < PROJECTOR_PIXEL_HEIGHT and
            col >= 0 and col < PROJECTOR_PIXEL_WIDTH):
            pixels[row, col] = GREEN
            pixels[row, col + 1] = GREEN
            pixels[row + 1, col] = GREEN
            pixels[row + 1, col + 1] = GREEN
    
    image = Image.fromarray(array(pixels, dtype=uint8), mode='RGB')
    image.save(filename, "png")
    print "Done."
    
    
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
        """ Calculate the dome projection parameters.  """
        # query the user for the projector resolution
        WIDTH, HEIGHT = get_projector_resolution()
        PROJECTOR_PIXEL_WIDTH = WIDTH
        PROJECTOR_PIXEL_HEIGHT = HEIGHT

        """ The first argument is the name of a file containing the list of
        centroids corresponding to the spots projected by the calibration
        device. """
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

        # Check the symmetry of these projector_points to check system
        # alignment.  Show the results and ask the user if they want to
        # continue with the calibration.
        check_point_symmetry(projector_points)
        continue_with_calibration = raw_input("Continue with calibration? (Y/N) ")
        if "y" not in continue_with_calibration.lower():
            sys.exit()

        # Search the parameter space to minimize differeces between measured
        # and calculated directions.
        parameters = estimate_parameters(projector_points)
        parameters['projector_pixel_width'] = PROJECTOR_PIXEL_WIDTH
        parameters['projector_pixel_height'] = PROJECTOR_PIXEL_HEIGHT

        # Save the centroid symmetry info, viewing direction errors, and
        # parameter values to a calibration_results file.
        """ Holding off on this for now. """

        # Print out the estimated parameter values
        print_parameters(parameters)

        # Generate an image that can be used to check the calibration.  It has
        # green spots that correspond to calibration_directions.
        datetime_string = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        save_calibration_image(parameters, datetime_string)






