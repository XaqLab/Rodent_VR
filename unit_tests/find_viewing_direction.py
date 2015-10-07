"""
This is the unit test for find_viewing_direction which calculates the direction
from the animal's location to a spot on the dome given the camera's direction
and the pixel coordinates of the spot in a photo taken with the camera.
Calls to this function look like this:

find_viewing_direction(photo_pixel, camera_direction, parameters)

where parameters is a dictionary that contains entries for animal_position,
dome_radius and dome_center.
"""
from matplotlib import pyplot as plot
from numpy import array, pi, arange, sin, cos, arcsin, arctan2, cross
from numpy.linalg import norm, inv
import random

# import from a relative directory
import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import find_viewing_direction
from dome_calibration import calc_distance_to_dome
from dome_calibration import rotate
import foscam_FI9821P as camera


DEBUG = False
required_accuracy = 1e-6

""" Pick a random camera orientation, then pick a point on the dome inside the
camera's field of view.  Calculate the location of the camera's
optical center and the direction from it to the point on the dome.  Use this
direction along with the camera's orientation to find the direction relative to
the camera's optical axis.  Use this relative direction and the camera's
calibration matrix to calculate the point's pixel coordinates. Call
find_viewing_direction and compare its results to the easily calculated
direction from the animal's location the point on the dome. """

# calculate the camera's field of view for later use
vector_to_upper_left_corner = inv(camera.matrix).dot([0.0, 0.0, 1.0])
vector_to_lower_right_corner = \
        inv(camera.matrix).dot([camera.pixel_width, camera.pixel_height, 1])
left = vector_to_upper_left_corner[0]
right = vector_to_lower_right_corner[0]
upper = vector_to_upper_left_corner[1]
lower = vector_to_lower_right_corner[1]

for i in range(1000):
    # pick a random radius for the dome
    radius = 0.2 + random.random()
    # pick a random location for the center of the dome
    x = random.random()
    y = random.random()
    z = random.random()
    center = array([x, y, z])
    # pick a random location inside the dome for the animal position
    r = 0.5 * radius * random.random()
    pitch = pi/2 * 2*(random.random() - 0.5)
    yaw = pi * 2*(random.random() - 0.5)
    animal_position = r * array([cos(pitch)*sin(yaw),
                                 cos(pitch)*cos(yaw),
                                 sin(pitch)])
    animal_position = animal_position + center
    # pick a random camera direction
    camera_pitch = pi/2 * 2*(random.random() - 0.5)
    camera_yaw = pi * 2*(random.random() - 0.5)
    camera_direction = array([cos(camera_pitch)*sin(camera_yaw),
                              cos(camera_pitch)*cos(camera_yaw),
                              sin(camera_pitch)])
    # Calculate the position of the camera's optical center. Find the unit
    # vectors of the camera's coordinate system and express
    # camera.reference_vector using those unit vectors.
    ccz = camera_direction
    ccx = cross(ccz, array([0.0, 0.0, 1.0]))
    ccx = ccx / norm(ccx)
    ccy = cross(ccz, ccx)
    reference_vector = (camera.reference_vector[0]*ccx +
                        camera.reference_vector[1]*ccy +
                        camera.reference_vector[2]*ccz)
    camera_focal_point = animal_position - reference_vector
    # pick a point on the dome inside the camera's field of view
    y = ccy * ((upper - lower) * random.random() + lower)
    x = ccx * ((right - left) * random.random() + left)
    direction_to_point = camera_direction + x + y
    direction_to_point = direction_to_point / norm(direction_to_point)
    parameters = dict(animal_position = animal_position,
                      dome_center = center,
                      dome_radius = radius)
    distance_to_point = calc_distance_to_dome(parameters, camera_focal_point,
                                              direction_to_point)
    camera_to_dome = distance_to_point * direction_to_point
    point_on_dome = camera_focal_point + camera_to_dome
    # calculate the animal's viewing direction, which is the result expected
    # from calc_viewing_direction
    expected_result = ((point_on_dome - animal_position)
                       / norm(point_on_dome - animal_position))
    # rotate camera_to_dome to put it in the camera's coordinate system
    yaw_rotation_vector = camera_yaw * array([0.0, 0.0, 1.0])
    camera_to_pixel = rotate(camera_to_dome, yaw_rotation_vector)
    pitch_rotation_vector = (pi/2 - camera_pitch) * array([1.0, 0.0, 0.0])
    camera_to_pixel = rotate(camera_to_pixel, pitch_rotation_vector)
    # use this vector and the camera matrix to find
    # the pixel coordinates of the spot on the dome
    pixel_vector = camera.matrix.dot(camera_to_pixel)
    x, y, w = pixel_vector
    photo_point = array([x, y]) / w
    if DEBUG:
        print
        print photo_point
    result = find_viewing_direction(photo_point, camera_direction, parameters)
    if norm(result - expected_result) > required_accuracy:
        print "Viewing direciton mismatch!"
        print "Expected result:", expected_result
        print "Actual result:  ", result

print "Done with find_viewing_direction test"
