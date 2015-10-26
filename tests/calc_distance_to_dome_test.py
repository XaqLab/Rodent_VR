"""
This is the unit test for calc_distance_to_dome which calculates the distance
to a sphere in a given direction from a position inside the sphere.  Calls to
this function look like this:

calc_distance_to_dome(parameters, position, direction)

where parameters is a dictionary that contains entries for dome_radius and
dome_center.
"""
from matplotlib import pyplot as plot
from numpy import array, pi, arange, sin, cos, sqrt, arcsin, arctan2
from numpy.linalg import norm
import random

# import from a relative directory
import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import calc_distance_to_dome


def test_calc_distance_to_dome():
    """ Place position at an arbitrary point in space and pick a random point
    on the sphere.  The distance between the points and the direction from one
    to the other can be easily calculated. """
    DEBUG = False
    required_accuracy = 1e-6
    parameters = dict(dome_center = [0, 0, 0], dome_radius = 1)
    for i in range(1000):
        # pick a random radius for the dome
        radius = random.random()
        # pick a random location for the center of the dome
        x = random.random()
        y = random.random()
        z = random.random()
        center = array([x, y, z])
        # pick a random position inside the dome
        r = radius * random.random()
        pitch = pi/2 * 2*(random.random() - 0.5)
        yaw = pi * 2*(random.random() - 0.5)
        position = r * array([cos(pitch)*sin(yaw),
                              cos(pitch)*cos(yaw),
                              sin(pitch)])
        position = position + center
        # pick a random position on the dome
        pitch = pi/2 * 2*(random.random() - 0.5)
        yaw = pi * 2*(random.random() - 0.5)
        point_on_dome = radius * array([cos(pitch)*sin(yaw),
                                        cos(pitch)*cos(yaw),
                                        sin(pitch)])
        point_on_dome = point_on_dome + center
        # calculate distance between points
        expected_result = norm(point_on_dome - position)
        # calculate direction from location to point on dome
        direction = (point_on_dome - position)/expected_result
        if DEBUG:
            x, y, z = direction
            pitch = arcsin(z)
            yaw = arctan2(x, y)
            print 180/pi*pitch, 180/pi*yaw
        # see what calc_distance_to_dome thinks
        parameters['dome_radius'] = radius
        parameters['dome_center'] = center
        result = calc_distance_to_dome(parameters, position, direction)
        assert norm(result - expected_result) < required_accuracy
        if DEBUG:
            print "Position:", position
            print "Point on dome:", point_on_dome
            print "Direction:", direction
            print "Expected result:", expected_result
            print "Actual result:  ", result
    
    
