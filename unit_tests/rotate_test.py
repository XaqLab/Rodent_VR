from numpy import pi, sin, cos, array, cross
from numpy.linalg import norm
import random

import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import rotate


def test_rotate_preserves_pixel_location():
    """ Check to see that the pixel location is preserved by the rotation """
    DEBUG = False
    required_accuracy = 1e-6

    for i in range(10):
        """ The optical_axis goes through the center of the image and
        vector_to_pixel points to a random pixel in the image.  The
        pixel, expressed in terms of row and column vectors, should be
        the same before and after rotation. """
        # find the pixel in camera coordinates where the camera's
        # optical axis is always the +z axis
        x = random.random() - 0.5
        y = random.random() - 0.5
        z = 1
        vector_to_pixel1 = array([x, y, z])
        optical_axis1 = array([0, 0, z])
        onscreen_vector1 = vector_to_pixel1 - optical_axis1
        col_vector1 = array([1, 0, 0])
        row_vector1 = array([0, 1, 0])
        col1 = onscreen_vector1.dot(col_vector1)
        row1 = onscreen_vector1.dot(row_vector1)
        pixel1 = array([col1, row1])
        # find the pixel in dome coordinates where the camera's optical
        # axis relative to the +y axis is specified by pitch and yaw
        pitch = pi/2*random.random()
        yaw = pi*(random.random() - 0.5)
        optical_axis2 = array([cos(pitch)*sin(yaw),
                               cos(pitch)*cos(yaw),
                               sin(pitch)])
        vector_to_pixel2 = rotate(vector_to_pixel1, array([pitch - pi/2, 0, 0]))
        vector_to_pixel2 = rotate(vector_to_pixel2, array([0, 0, -yaw]))
        onscreen_vector2 = vector_to_pixel2 - optical_axis2
        col_vector2 = cross(optical_axis2, array([0, 0, 1]))
        col_vector2 = col_vector2/norm(col_vector2)
        row_vector2 = cross(optical_axis2, array(col_vector2))
        row_vector2 = row_vector2/norm(row_vector2)
        col2 = onscreen_vector2.dot(col_vector2)
        row2 = onscreen_vector2.dot(row_vector2)
        pixel2 = array([col2, row2])
        assert norm(pixel2 - pixel1) < required_accuracy
        if DEBUG:
            print "Mismatch of:", norm(pixel2 - pixel1)
            print "Pitch and yaw are:", 180/pi*pitch, 180/pi*yaw
            print "Column and row before rotation: %10f, %10f" % (col1, row1)
            print "Column and row after rotation:  %10f, %10f" % (col2, row2)
    
