"""
This is the unit test for calc_viewing_directions which returns the viewing
directions from the animal's position for spots in four, overlapping
calibration photos.  Each photo contains 4 spots, one in each quadrant: upper
left, upper right, lower left, and lower right.  The spots on the dome are in a
3 by 3 grid so some of the spots appear in more than one photo.
Calls to this function look like this:

calc_viewing_directions(photo_points, parameters)

photo_points is a list of four lists which contain the (u,v) coordinates for
the centers of the four spots in each calibration photo.  The lists are in this
order: top left photo, top right photo, bottom left photo, bottom right photo

parameters is a dictionary that contains entries for animal_position,
dome_radius and dome_center.
"""
from numpy import array, pi, sin, cos, arcsin, arctan2, cross, zeros
from numpy import uint8
from numpy.linalg import norm
import random
from PIL import Image

# import from a relative directory
import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import rotate
from dome_calibration import calc_distance_to_dome
from dome_calibration import calc_viewing_directions
import foscam_FI9821P as camera


class CalibrationPoint():
    def __init__(self, parameters):
        """ keep track of a single calibration point on the dome and make it
        easy to move around """
        # default point is the point on the dome that is in the 0 pitch, 0 yaw
        # direction from the animal's position
        self.direction = array([0.0, 1.0, 0.0])
        animal_position = parameters['animal_position']
        distance_to_dome = calc_distance_to_dome(parameters, animal_position,
                                                 self.direction)
        self.point = animal_position + distance_to_dome * self.direction


    def rotate_pitch_and_yaw(self, pitch, yaw, parameters):
        x, y, z = self.direction
        new_pitch = arcsin(z) + pitch
        new_yaw = arctan2(x, y) + yaw
        self.direction = array([cos(new_pitch)*sin(new_yaw),
                                cos(new_pitch)*cos(new_yaw),
                                sin(new_pitch)])
        animal_position = parameters['animal_position']
        distance_to_dome = calc_distance_to_dome(parameters, animal_position,
                                                 self.direction)
        self.point = animal_position + distance_to_dome * self.direction


    def move_up(self, parameters):
        """ increase pitch by a random amount """
        pitch = 5*pi/180*random.random()
        self.rotate_pitch_and_yaw(pitch, 0, parameters)


    def move_down(self, parameters):
        """ decrease pitch by a random amount """
        pitch = 5*pi/180*random.random()
        self.rotate_pitch_and_yaw(-pitch, 0, parameters)


    def move_right(self, parameters):
        """ increase yaw by a random amount """
        yaw = 10*pi/180*random.random()
        self.rotate_pitch_and_yaw(0, yaw, parameters)


    def move_left(self, parameters):
        """ decrease yaw by a random amount """
        yaw = 10*pi/180*random.random()
        self.rotate_pitch_and_yaw(0, -yaw, parameters)


    def move_to_upper_left(self, photo, parameters):
        u, v = photo.point(self, parameters)
        while u < 0:
            # move the point right
            self.move_right(parameters)
            u, v = photo.point(self, parameters)
        while u > 0.25*camera.pixel_width:
            # move the point left
            self.move_left(parameters)
            u, v = photo.point(self, parameters)
        while v < 0:
            # move the point down
            self.move_down(parameters)
            u, v = photo.point(self, parameters)
        while v > 0.25*camera.pixel_height:
            # move the point up
            self.move_up(parameters)
            u, v = photo.point(self, parameters)


    def move_to_upper_right(self, photo, parameters):
        u, v = photo.point(self, parameters)
        while u < 0.75*camera.pixel_width:
            # move the point right
            self.move_right(parameters)
            u, v = photo.point(self, parameters)
        while u > camera.pixel_width:
            # move the point left
            self.move_left(parameters)
            u, v = photo.point(self, parameters)
        while v < 0:
            # move the point down
            self.move_down(parameters)
            u, v = photo.point(self, parameters)
        while v > 0.25*camera.pixel_height:
            # move the point up
            self.move_up(parameters)
            u, v = photo.point(self, parameters)


    def move_to_lower_left(self, photo, parameters):
        u, v = photo.point(self, parameters)
        while u < 0:
            # move the point right
            self.move_right(parameters)
            u, v = photo.point(self, parameters)
        while u > 0.25*camera.pixel_width:
            # move the point left
            self.move_left(parameters)
            u, v = photo.point(self, parameters)
        while v < 0.75*camera.pixel_height:
            # move the point down
            self.move_down(parameters)
            u, v = photo.point(self, parameters)
        while v > camera.pixel_height:
            # move the point up
            self.move_up(parameters)
            u, v = photo.point(self, parameters)


    def move_to_lower_right(self, photo, parameters):
        u, v = photo.point(self, parameters)
        while u < 0.75*camera.pixel_width:
            # move the point right
            self.move_right(parameters)
            u, v = photo.point(self, parameters)
        while u > camera.pixel_width:
            # move the point left
            self.move_left(parameters)
            u, v = photo.point(self, parameters)
        while v < 0.75*camera.pixel_height:
            # move the point down
            self.move_down(parameters)
            u, v = photo.point(self, parameters)
        while v > camera.pixel_height:
            # move the point up
            self.move_up(parameters)
            u, v = photo.point(self, parameters)


class CalibrationPoints():
    def __init__(self, parameters):
        """ keep track of the 3 by 3 grid of points on the dome used for
        taking calibration photos """
        self.upper_left = CalibrationPoint(parameters)
        self.upper = CalibrationPoint(parameters)
        self.upper_right = CalibrationPoint(parameters)
        self.left = CalibrationPoint(parameters)
        self.center = CalibrationPoint(parameters)
        self.right = CalibrationPoint(parameters)
        self.lower_left = CalibrationPoint(parameters)
        self.lower = CalibrationPoint(parameters)
        self.lower_right = CalibrationPoint(parameters)


class CameraOrientation():
    def __init__(self, pitch=0, yaw=0):
        """ keep track of the camera orientation for a calibration photo and
        make it easy adjust the camera orientation and check the pixel
        coordinates of points in the camera's field of view """
        self.pitch = pitch
        self.yaw = yaw
        self.direction = array([cos(self.pitch)*sin(self.yaw),
                                cos(self.pitch)*cos(self.yaw),
                                sin(self.pitch)])


    def rotate_pitch_and_yaw(self, pitch, yaw):
        self.pitch += pitch
        self.yaw += yaw
        self.direction = array([cos(self.pitch)*sin(self.yaw),
                                cos(self.pitch)*cos(self.yaw),
                                sin(self.pitch)])


    def increase_pitch(self):
        """ increase pitch by a random amount less than half of the camera's
        vertical field of view """
        pitch = 5*pi/180*random.random()
        self.rotate_pitch_and_yaw(pitch, 0)


    def decrease_pitch(self):
        """ decrease pitch by a random amount less than half of the camera's
        vertical field of view """
        pitch = 5*pi/180*random.random()
        self.rotate_pitch_and_yaw(-pitch, 0)


    def increase_yaw(self):
        """ increase yaw by a random amount less than half of the camera's
        horizontal field of view """
        yaw = 10*pi/180*random.random()
        self.rotate_pitch_and_yaw(0, yaw)


    def decrease_yaw(self):
        """ decrease yaw by a random amount less than half of the camera's
        horizontal field of view """
        yaw = 10*pi/180*random.random()
        self.rotate_pitch_and_yaw(0, -yaw)


    def move_to_upper_left(self, point, parameters):
        u, v = self.point(point, parameters)
        while u < 0:
            # rotate the camera left
            self.decrease_yaw()
            u, v = self.point(point, parameters)
        while u > 0.25*camera.pixel_width:
            # rotate the camera right
            self.increase_yaw()
            u, v = self.point(point, parameters)
        while v < 0:
            # rotate the camera up
            self.increase_pitch()
            u, v = self.point(point, parameters)
        while v > 0.25*camera.pixel_height:
            # rotate the camera down
            self.decrease_pitch()
            u, v = self.point(point, parameters)


    def move_to_upper_right(self, point, parameters):
        u, v = self.point(point, parameters)
        while u < 0.75*camera.pixel_width:
            # rotate the camera left
            self.decrease_yaw()
            u, v = self.point(point, parameters)
        while u > camera.pixel_width:
            # rotate the camera right
            self.increase_yaw()
            u, v = self.point(point, parameters)
        while v < 0:
            # rotate the camera up
            self.increase_pitch()
            u, v = self.point(point, parameters)
        while v > 0.25*camera.pixel_height:
            # rotate the camera down
            self.decrease_pitch()
            u, v = self.point(point, parameters)


    def move_to_lower_left(self, point, parameters):
        u, v = self.point(point, parameters)
        while u < 0:
            # rotate the camera left
            self.decrease_yaw()
            u, v = self.point(point, parameters)
        while u > 0.25*camera.pixel_width:
            # rotate the camera right
            self.increase_yaw()
            u, v = self.point(point, parameters)
        while v < 0.75*camera.pixel_height:
            # rotate the camera up
            self.increase_pitch()
            u, v = self.point(point, parameters)
        while v > camera.pixel_height:
            # rotate the camera down
            self.decrease_pitch()
            u, v = self.point(point, parameters)


    def move_to_lower_right(self, point, parameters):
        u, v = self.point(point, parameters)
        while u < 0.75*camera.pixel_width:
            # rotate the camera left
            self.decrease_yaw()
            u, v = self.point(point, parameters)
        while u > camera.pixel_width:
            # rotate the camera right
            self.increase_yaw()
            u, v = self.point(point, parameters)
        while v < 0.75*camera.pixel_height:
            # rotate the camera up
            self.increase_pitch()
            u, v = self.point(point, parameters)
        while v > camera.pixel_height:
            # rotate the camera down
            self.decrease_pitch()
            u, v = self.point(point, parameters)


    def calc_opencv_unit_vectors(self):
        """ calculate the unit vectors for OpenCV's camera coordinate system """
        dome_z = array([0.0, 0.0, 1.0])
        camera_z = self.direction
        camera_x = cross(camera_z, dome_z)
        camera_x = camera_x / norm(camera_x)
        camera_y = cross(camera_z, camera_x)
        return camera_x, camera_y, camera_z
    
    
    def animal_to_camera(self):
        """ convert the vector that points from the camera's optical center to the
        animal's position from camera coordinates into dome coordinates, then
        multiply by -1 so that it points from the animal's position to the
        camera's optical center """
        camera_x, camera_y, camera_z = self.calc_opencv_unit_vectors()
        reference_vector = (camera.reference_vector[0]*camera_x +
                            camera.reference_vector[1]*camera_y +
                            camera.reference_vector[2]*camera_z)
        return -reference_vector


    def point(self, point_on_dome, parameters):
        """ return the (u, v) coordinates of point_on_dome when
        photographed with the camera in this orientation """
        # calculate vector from camera's optical center to point_on_dome
        animal_position = parameters['animal_position']
        camera_focal_point = animal_position + self.animal_to_camera()
        camera_to_dome = point_on_dome.point - camera_focal_point
        # rotate camera_to_dome so that it is in OpenCV coordinates
        yaw_rotation_vector = self.yaw * array([0.0, 0.0, 1.0])
        camera_to_dome = rotate(camera_to_dome, yaw_rotation_vector)
        pitch_rotation_vector = (pi/2 - self.pitch) * array([1.0, 0.0, 0.0])
        camera_to_dome = rotate(camera_to_dome, pitch_rotation_vector)
        # use camera.matrix to calculate the point's coordinates
        point_vector = camera.matrix.dot(camera_to_dome)
        u, v, w = point_vector
        point = array([u, v]) / w
        return point


class CameraOrientations():
    def __init__(self, pitch=0, yaw=0):
        """ keep track of all the camera orientations used for taking
        calibration photos """
        self.upper_left = CameraOrientation()
        self.upper_right = CameraOrientation()
        self.lower_left = CameraOrientation()
        self.lower_right = CameraOrientation()


def generate_directions_and_points(parameters):
    """ Generate photo_points argument to test calc_viewing_directions.
    The arrangement of the spots on the dome looks like this:
        upper_left          upper           upper_right 
        left                center          right
        lower_left          lower           lower_right """
    DEBUG = False

    photos = CameraOrientations()
    spots = CalibrationPoints(parameters)
    """ Find an orientation for the lower_left photo """
    # adjust the camera orientation for the lower left photo until the 0 pitch,
    # 0 yaw point is in the lower left quadrant
    photos.lower_left.move_to_lower_left(spots.lower_left, parameters)
    """ Find a position for the center calibration point """
    # adjust the position of the center calibration point until it's in the
    # upper right quadrant of the lower left photo
    spots.center.move_to_upper_right(photos.lower_left, parameters)
    """ Find an orientation for the upper_left photo """
    # adjust the camera orientation for the upper left photo until the center
    # point is in the lower right quadrant
    photos.upper_left.move_to_lower_right(spots.center, parameters)
    """ Find an orientation for the upper_right photo """
    # adjust the camera orientation for the upper right photo until the center
    # point is in the lower left quadrant
    photos.upper_right.move_to_lower_left(spots.center, parameters)
    """ Find an orientation for the lower_right photo """
    # adjust the camera orientation for the lower right photo until the center
    # point is in the upper left quadrant
    photos.lower_right.move_to_upper_left(spots.center, parameters)
    """ Find a position for the upper_left calibration point """
    # adjust the position of the upper_left calibration point until it's in the
    # upper left quadrant of the upper left photo
    spots.upper_left.move_to_upper_left(photos.upper_left, parameters)
    """ Find a position for the upper calibration point """
    # adjust the position of the upper calibration point until it's in the
    # upper right quadrant of the upper left photo
    spots.upper.move_to_upper_right(photos.upper_left, parameters)
    # adjust the position of the upper calibration point until it's in the
    # upper left quadrant of the upper right photo
    spots.upper.move_to_upper_left(photos.upper_right, parameters)
    """ Find a position for the upper_right calibration point """
    # adjust the position of the upper_right calibration point until it's in
    # the upper right quadrant of the upper right photo
    spots.upper_right.move_to_upper_right(photos.upper_right, parameters)
    """ Find a position for the left calibration point """
    # adjust the position of the left calibration point until it's in the
    # lower left quadrant of the upper left photo
    spots.left.move_to_lower_left(photos.upper_left, parameters)
    # adjust the position of the left calibration point until it's in the
    # upper left quadrant of the lower left photo
    spots.left.move_to_upper_left(photos.lower_left, parameters)
    """ Find a position for the right calibration point """
    # adjust the position of the right calibration point until it's in the
    # lower right quadrant of the upper right photo
    spots.right.move_to_lower_right(photos.upper_right, parameters)
    # adjust the position of the right calibration point until it's in the
    # upper right quadrant of the lower right photo
    spots.right.move_to_upper_right(photos.lower_right, parameters)
    """ Find a position for the lower_left calibration point """
    # adjust the position of the lower_left calibration point until it's in the
    # lower left quadrant of the lower left photo
    spots.lower_left.move_to_lower_left(photos.lower_left, parameters)
    """ Find a position for the lower calibration point """
    # adjust the position of the lower calibration point until it's in the
    # lower right quadrant of the lower left photo
    spots.lower.move_to_lower_right(photos.lower_left, parameters)
    # adjust the position of the lower calibration point until it's in the
    # lower left quadrant of the lower right photo
    spots.lower.move_to_lower_left(photos.lower_right, parameters)
    """ Find a position for the lower_right calibration point """
    # adjust the position of the lower_right calibration point until it's in
    # the lower right quadrant of the lower right photo
    spots.lower_right.move_to_lower_right(photos.lower_right, parameters)
    if DEBUG:
        print "Camera Orientations, left to right, top to bottom"
        print 180/pi*photos.upper_left.pitch, 180/pi*photos.upper_left.yaw
        print 180/pi*photos.upper_right.pitch, 180/pi*photos.upper_right.yaw
        print 180/pi*photos.lower_left.pitch, 180/pi*photos.lower_left.yaw
        print 180/pi*photos.lower_right.pitch, 180/pi*photos.lower_right.yaw
    """ Calculate the viewing direction for each point """
    animal_position = parameters['animal_position']
    upper_left = spots.upper_left.point - animal_position
    upper_left = upper_left / norm(upper_left)
    upper = spots.upper.point - animal_position
    upper = upper / norm(upper)
    upper_right = spots.upper_right.point - animal_position
    upper_right = upper_right / norm(upper_right)
    left = spots.left.point - animal_position
    left = left / norm(left)
    center = spots.center.point - animal_position
    center = center / norm(center)
    right = spots.right.point - animal_position
    right = right / norm(right)
    lower_left = spots.lower_left.point - animal_position
    lower_left = lower_left / norm(lower_left)
    lower = spots.lower.point - animal_position
    lower = lower / norm(lower)
    lower_right = spots.lower_right.point - animal_position
    lower_right = lower_right / norm(lower_right)
    directions = [upper_left,  upper, upper_right,
                        left, center,       right,
                  lower_left,  lower, lower_right]
    """ Calculate the point coordinates for each photo """
    upper_left_photo = \
            [photos.upper_left.point(spots.upper_left, parameters),
             photos.upper_left.point(spots.upper, parameters),
             photos.upper_left.point(spots.left, parameters),
             photos.upper_left.point(spots.center, parameters)];
    upper_right_photo = \
            [photos.upper_right.point(spots.upper, parameters),
             photos.upper_right.point(spots.upper_right, parameters),
             photos.upper_right.point(spots.center, parameters),
             photos.upper_right.point(spots.right, parameters)];
    lower_left_photo = \
            [photos.lower_left.point(spots.left, parameters),
             photos.lower_left.point(spots.center, parameters),
             photos.lower_left.point(spots.lower_left, parameters),
             photos.lower_left.point(spots.lower, parameters)];
    lower_right_photo = \
            [photos.lower_right.point(spots.center, parameters),
             photos.lower_right.point(spots.right, parameters),
             photos.lower_right.point(spots.lower, parameters),
             photos.lower_right.point(spots.lower_right, parameters)];
    photo_points = [upper_left_photo, upper_right_photo,
                    lower_left_photo, lower_right_photo];
    return directions, photo_points


def show_calibration_photo(point_list):
    """ for debugging purposes """
    pixels = zeros([camera.pixel_height, camera.pixel_width], dtype=uint8)
    for point in point_list:
        u, v = point
        row = int(round(v))
        col = int(round(u))
        pixels[row,col] = 255
        image = Image.fromarray(pixels, mode='L')
    image.show()


def test_calc_viewing_directions():
    """ 
    Keep adjusting the camera orientation for the lower left photo until the point
    on the dome at 0 pitch, 0 yaw is in the lower left quandrant.
    
    Then add a new point on the dome and move it until it's in the upper right
    quandrant of the photo.  This point appears in all 4 photos.
    
    Now pick the orientations for the other 3 photos by moving them until this
    point is in the appropriate quadrant for each photo.
    
    Then add 7 more points and move each around until it's in the correct
    quadrant of one or more photos.

    Finally find the (u, v) coordinates of the four points in each of the four
    calibration photos and pass them (as photo_points) to
    calc_viewing_directions to confirm that it calculates the correct viewing
    directions.
    """
    DEBUG = False
    required_accuracy = 1e-6
    for i in range(10):
        # pick a random radius for the dome, above some minimum value
        radius = 0.2 + random.random()
        # pick a random location for the center of the dome
        x = 0  # the dome center is in the y-z plane
        y = random.random()
        z = random.random()
        center = array([x, y, z])
        # pick a random location inside the dome for the animal position
        r = 0.5 * radius * random.random()
        pitch = pi/2 * 2*(random.random() - 0.5)
        yaw = 0  # the animal position is in the y-z plane
        animal_position = r * array([cos(pitch)*sin(yaw),
                                     cos(pitch)*cos(yaw),
                                     sin(pitch)])
        animal_position = animal_position + center
        # create the parameters dictionary
        parameters = dict(animal_position = animal_position,
                          dome_center = center,
                          dome_radius = radius)
        # generate viewing directions and photo points that are consistent with the
        # camera calibration parameters
        expected_result, photo_points = generate_directions_and_points(parameters)
        if DEBUG:
            for r in expected_result:
                x, y, z = r
                pitch = 180/pi*arcsin(z)
                yaw = 180/pi*arctan2(x,y)
                print r, pitch, yaw
            print
            for photo in photo_points:
                for point in photo:
                    print point
                print
                show_calibration_photo(photo)
        # see what calc_viewing_directions gives us
        result = calc_viewing_directions(photo_points, parameters)
        for i in range(len(expected_result)):
            assert norm(result[i] - expected_result[i]) < required_accuracy
            if DEBUG:
                print "Expected direction:", expected_result[i]
                print "Actual direction:", result[i]
        
