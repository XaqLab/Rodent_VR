"""
This is the unit test for camera_orientation_error which is minimized in order
to infer the camera's orientation from a photo containing a point with a known
viewing direction from the animal's position.
Calls to this function look like this:

camera_orientation_error(orientation, lower_left_pixel, known_direction,
                         parameters)

where parameters is a dictionary that contains entries for animal_position,
dome_radius and dome_center.
"""
from numpy import array, pi, sin, cos, arcsin, arctan2, cross
from numpy.linalg import norm, inv
import random
from scipy.optimize import minimize

# import from a relative directory
import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import camera_orientation_error
from dome_calibration import find_viewing_direction
from dome_calibration import calc_distance_to_dome
from dome_calibration import rotate
import foscam_FI9821P as camera


def test_camera_orientation_error():
    """ Pick a random camera orientation, then pick a point on the dome inside
    the lower left quadrant of the camera's field of view.  Calculate the
    location of the camera's optical center and the direction from it to the
    point on the dome.  Use this direction along with the camera's orientation
    to find the point's pixel coordinates. Minimize camera_orientation_error to
    find the camera's orientation and compare it to the randomly chosen
    orientation. """
    DEBUG = False
    required_accuracy = 1e-6

    # calculate the camera's field of view for later use
    vector_to_upper_left_corner = inv(camera.matrix).dot([0.0, 0.0, 1.0])
    vector_to_lower_right_corner = \
            inv(camera.matrix).dot([camera.pixel_width, camera.pixel_height, 1])
    left = vector_to_upper_left_corner[0]
    right = vector_to_lower_right_corner[0]
    upper = vector_to_upper_left_corner[1]
    lower = vector_to_lower_right_corner[1]
    
    for i in range(100):
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
        # limit pitch to +/- 60 degrees to avoid the ambiguity caused by pitch
        # angles near 90 degrees
        camera_pitch = pi/3 * 2*(random.random() - 0.5)
        camera_yaw = pi * 2*(random.random() - 0.5)
        camera_direction = array([cos(camera_pitch)*sin(camera_yaw),
                                  cos(camera_pitch)*cos(camera_yaw),
                                  sin(camera_pitch)])
        expected_result = camera_direction
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
        y = ccy * ((0 - lower) * random.random() + lower)
        x = ccx * ((0 - left) * random.random() + left)
        direction_to_point = camera_direction + x + y
        direction_to_point = direction_to_point / norm(direction_to_point)
        parameters = dict(animal_position = animal_position,
                          dome_center = center,
                          dome_radius = radius)
        distance_to_point = calc_distance_to_dome(parameters, camera_focal_point,
                                                  direction_to_point)
        camera_to_dome = distance_to_point * direction_to_point
        point_on_dome = camera_focal_point + camera_to_dome
        # calculate the animal's viewing direction
        known_direction = ((point_on_dome - animal_position)
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
        lower_left_point = array([x, y]) / w
        # estimate a starting orientation
        [x, y, z] = known_direction
        known_pitch = arcsin(z)
        known_yaw = arctan2(x, y)
        apparent_direction = find_viewing_direction(lower_left_point,
                                                    array([0, 1, 0]),
                                                    parameters)
        [x, y, z] = apparent_direction
        apparent_pitch = arcsin(z)
        apparent_yaw = arctan2(x, y)
        starting_pitch = known_pitch - apparent_pitch
        starting_yaw = known_yaw - apparent_yaw
        starting_direction = array([cos(starting_pitch)*sin(starting_yaw),
                                    cos(starting_pitch)*cos(starting_yaw),
                                    sin(starting_pitch)])
        starting_orientation = (starting_pitch, starting_yaw)
        args = (lower_left_point, known_direction, parameters)
        results = minimize(camera_orientation_error, starting_orientation,
                           args=args, method='Nelder-Mead', options={'xtol':1e-9})
        estimated_pitch, estimated_yaw = results['x']
        estimated_direction = array([cos(estimated_pitch)*sin(estimated_yaw),
                                     cos(estimated_pitch)*cos(estimated_yaw),
                                     sin(estimated_pitch)])
        result = estimated_direction
        assert norm(result - expected_result) < required_accuracy
        if DEBUG:
            print "Starting pitch, yaw:", 180/pi*starting_pitch, 180/pi*starting_yaw
            print "Estimated pitch, yaw:", 180/pi*estimated_pitch, 180/pi*estimated_yaw
            print "Camera pitch, yaw:   ", 180/pi*camera_pitch, 180/pi*camera_yaw
            print "Pitch error:", 180/pi*(estimated_pitch - camera_pitch)
            print "Yaw error:", 180/pi*(estimated_yaw - camera_yaw)
            print "Final function value:", results['fun']
            print lower_left_point
            print "Camera direction   ", camera_direction
            print "Estimated direction", estimated_direction
            print "Viewing direction using camera direction:   ",
            print find_viewing_direction(lower_left_point, camera_direction,
                                         parameters)
            print "Viewing direction using estimated direction:",
            print find_viewing_direction(lower_left_point, estimated_direction,
                                         parameters)
    
