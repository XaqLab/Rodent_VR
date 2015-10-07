import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi, sin, cos, array, sqrt, cross
from numpy.linalg import norm
import matplotlib.pyplot as plt
import random

mpl.rcParams['legend.fontsize'] = 10

import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import rotate


required_accuracy = 0.000001


""" Check to see that the pixel location is preserved by the rotation """
for i in range(100):
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
    if norm(pixel2 - pixel1) > required_accuracy:
        print "Mismatch of:", norm(pixel2 - pixel1)
        print "Pitch and yaw are:", 180/pi*pitch, 180/pi*yaw
        print "Column and row before rotation: %10f, %10f" % (col1, row1)
        print "Column and row after rotation:  %10f, %10f" % (col2, row2)
print "Done with rotation test"


""" Check that rotating a vector by pitch and yaw angles and then rotating it
again by -pitch and -yaw recovers the original vector """
"""
for i in range(10):
    x = random.random()
    y = random.random()
    z = random.random()
    vector1 = array([x, y, z])
    pitch = pi/2*random.random()
    yaw = pi*(random.random() - 0.5)
    vector2 = rotate(rotate(vector1, pitch, yaw), -pitch, -yaw)
    vector_to_pixel2 = rotate(vector_to_pixel1, array([pitch - pi/2, 0, 0]))
    vector_to_pixel2 = rotate(vector_to_pixel1, array([0, 0, -yaw]))
    if norm(vector2 - vector1) > required_accuracy:
        print "Mismatch of:", norm(vector2 - vector1)
        print "Pitch and yaw are:", 180/pi*pitch, 180/pi*yaw
        print "vector before rotation:", vector1
        print "vector after rotation: ", vector2
"""


""" code for plotting 3D vectors
fig = plt.figure()
ax = fig.gca(projection='3d')
# plot x, y, z unit vectors in red, green, and blue respectively to help orient
# the coordinate system
vector = array([1, 0, 0])
x = array([0, vector[0]])
y = array([0, vector[1]])
z = array([0, vector[2]])
ax.plot(x, y, z, 'r')
vector = array([0, 1, 0])
x = array([0, vector[0]])
y = array([0, vector[1]])
z = array([0, vector[2]])
ax.plot(x, y, z, 'g')
vector = array([0, 0, 1])
x = array([0, vector[0]])
y = array([0, vector[1]])
z = array([0, vector[2]])
ax.plot(x, y, z, 'b')

# now plot a vector and then rotate it
pitch = 30*pi/180
yaw = 45*pi/180
vector = array([cos(pitch)*sin(yaw),
                cos(pitch)*cos(yaw),
                sin(pitch)])
print vector
x = array([0, vector[0]])
y = array([0, vector[1]])
z = array([0, vector[2]])
ax.plot(x, y, z)

vector = rotate(array([1/sqrt(2), 1/sqrt(2), 0]), pitch, 0)
print vector
x = array([0, vector[0]])
y = array([0, vector[1]])
z = array([0, vector[2]])
ax.plot(x, y, z)

ax.legend()
plt.show()
"""
