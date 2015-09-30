"""
This is the script I wrote to characterize the camera used to estimate the dome
projection geometry.  I displayed a black and white checkerboard pattern
16 squares wide and 9 squares high, matching the TV's 16:9 aspect ratio.
I took 10 pictures of this pattern with the camera, moving the camera between
pictures.
"""
from numpy import array, zeros, float32, cross
from numpy import pi, arcsin
from numpy.linalg import norm
from scipy.optimize import minimize
import cv2  # open computer vision library
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


CAMERA_RESOLUTION = (1280, 720)  # (x, y)
BOARD_SIZE = (15, 8)  # (x, y)
DEBUG = False


"""
Prepare points for the checker board's "internal" corners.  The checker board has
9 rows and 16 columns, this results in 15 "internal" corners per row and 8 per
column.  The measured viewing area on the TV I used for calibration was 882 mm
wide by 488 mm high.  
"""
dx = 0.882/16
dy = 0.488/9
checker_board = array([[x*dx, y*dy, 0]
                       for y in range(BOARD_SIZE[1])
                       for x in range(BOARD_SIZE[0])], float32)


def plot_points(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(points)):
        a = row_vector(points[i])
        x = array([a[0]])
        y = array([a[1]])
        z = array([a[2]])
        ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def plot_vectors(starting_points, ending_points):
    assert len(starting_points) == len(ending_points)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(starting_points)):
        a = row_vector(starting_points[i])
        b = row_vector(ending_points[i])
        x = array([a[0], b[0]])
        y = array([a[1], b[1]])
        z = array([a[2], b[2]])
        ax.plot(x, y, z)
    plt.show()


def plot_vectors_from_center(center_point, ending_points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a = row_vector(center_point)
    for i in range(len(ending_points)):
        b = row_vector(ending_points[i])
        x = array([a[0], b[0]])
        y = array([a[1], b[1]])
        z = array([a[2], b[2]])
        ax.plot(x, y, z)
    plt.show()


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


def row_vector(vector):
    """
    take a list or array and make sure it's in row vector format for numpy
    """
    shape = array(vector).shape
    assert len(shape) < 3, "Input vector must have less than three dimensions."
    if len(shape) > 1:
        assert 1 in shape
    length = len(vector)
    if shape == (length, 1):
        # input is a column vector
        row_vector = array([i[0] for i in vector])
    else:
        # input is a row vector
        row_vector = array(vector)
    return row_vector


def find_rotation_vector(vector1, vector2):
    """
    find the rotation vector, V, such that
    vector2 = cv2.Rodrigues(V)[0].dot(vector1) 
    """
    assert len(vector1) == 3, "vector1 must have exactly 3 elements"
    assert len(vector2) == 3, "vector2 must have exactly 3 elements"
    row_v1 = row_vector(vector1)
    row_v2 = row_vector(vector2)
    unit_v1 = row_v1 / norm(row_v1)
    unit_v2 = row_v2 / norm(row_v2)
    if unit_v1.dot(unit_v2) == 1:
        # vector1 and vector2 are identical
        rotation_vector = column_vector(zeros(3))
    else:
        cross_product = cross(unit_v1, unit_v2)
        sin_of_angle = norm(cross_product)
        cos_of_angle = unit_v1.dot(unit_v2)
        if cos_of_angle > 0:
            angle = arcsin(sin_of_angle)
        else:
            angle = pi - arcsin(sin_of_angle)
        rotation_vector = angle*cross_product/sin_of_angle
    return column_vector(rotation_vector)


def find_image_points(image_files):
    """
    Return a list of arrays containing the coordinates of the checker board's
    "internal" corners in each image.
    """
    files_used = []    # files in which the checker board was found
    image_points = []  # 2d points in image plane.
    for filename in filenames:
        image = cv2.imread(filename)
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert the gray scale image because findChessboardCorners requires a
        # white background.
        inverted_image = 255 - gray_scale_image
        assert inverted_image.shape[::-1] == CAMERA_RESOLUTION
    
        # Find the chess board corners
        found, corners = cv2.findChessboardCorners(inverted_image,
                                                   BOARD_SIZE,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                   cv2.CALIB_CB_ASYMMETRIC_GRID)
    
        # If found add corners to image points
        if found == True:
            files_used.append(filename)
            image_points.append(corners)
    
            if DEBUG:
                # Show each image with the corners drawn on
                cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
                cv2.imshow('Corners', image)
                cv2.waitKey(500)
        else:
            print "No checker board found in", filename
    
    if DEBUG:
        for i in range(len(files_used)):
            print
            print files_used[i]
            print "Rotation:"
            print rotation_vectors[i]
            print 180/pi*norm(rotation_vectors[i])
            print "Translation:"
            print translation_vectors[i]
        cv2.destroyAllWindows()

    return image_points


def find_object_positions_and_orientations(filenames):
    """
    Return lists of rotation and translation vectors that describe the checker
    board's location and orientation for each image in filenames.
    """
    files_used = []    # files in which the checker board was found
    object_positions = []
    object_orientations = []
    for filename in filenames:
        image = cv2.imread(filename)
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Invert the gray scale image because findChessboardCorners requires a
        # white background.
        inverted_image = 255 - gray_scale_image
    
        # Find the chess board corners
        found, corners = cv2.findChessboardCorners(inverted_image,
                                                   BOARD_SIZE,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                   cv2.CALIB_CB_ASYMMETRIC_GRID)
    
        # If found find object pose
        if found == True:
            files_used.append(filename)
            outputs = cv2.solvePnP(checker_board, corners,
                                   camera_matrix, distortion_coefficients)
            retval, rotation_vector, translation_vector = outputs
            object_positions.append(translation_vector)
            object_orientations.append(rotation_vector)
    
            # Draw and display the corners
            if DEBUG:
                cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
                cv2.imshow('Corners', image)
                cv2.waitKey(500)
        else:
            print "Checker board not found in", filename
    
    if DEBUG:
        cv2.destroyAllWindows()
    
        for i in range(len(files_used)):
            print
            print files_used[i]
            print "Positions:"
            print object_positions[i]
            print "Orientation:"
            print object_orientations[i]
            print 180/pi*norm(object_orientations[i])
    return array(object_positions), array(object_orientations)


def find_camera_positions_and_orientations(camera_matrix, object_positions,
                                           object_orientations, verbose=False):
    """
    Convert object positions and orientations in camera coordinates to camera
    positions and orientations in checker board coordinates by assuming that
    the checker board is stationary and only the camera moves between photos.
    """
    # The object position vectors point from the camera's optical center to the
    # origin of the checker board coordinate system.  In addition to changing
    # the direction of this vector to point from the checker board origin to
    # the optical center, it is also necessary to consider the rotation of the
    # checker board to convert the vector to checker board coordinates.
    camera_positions = zeros(object_positions.shape)
    camera_orientations = zeros(object_orientations.shape)
    for i in range(len(object_positions)):
        rotation_matrix = cv2.Rodrigues(-object_orientations[i])[0]
        camera_positions[i] = rotation_matrix.dot(-object_positions[i])

        # Find the rotation matrix that converts the direction to the object's
        # origin to the direction of the camera's optical axis in the camera's
        # coordinate system.
        optical_axis = [0, 0, 1]
        rotation_vector = find_rotation_vector(object_positions[i],
                                               optical_axis)
        # Determine the camera's orientation by rotating from the direction to
        # the object origin to the direction of the camera's optical axis in
        # the object coordinate system.
        camera_orientations[i] = -object_orientations[i]
    return camera_positions, camera_orientations


def point_to_positions(point, camera_positions, verbose=False):
    """
    Calculate the distances between point and each position of the camera's
    optical center.
    """
    # Calculate and return distances
    distances = array([norm(c - column_vector(point))
                       for c in camera_positions])
    if verbose:
        print distances
        print "mean distance:", distances.mean()
        print "standard deviation of distance:", distances.std()
        print "point:", point
    return distances
    

def point_to_positions_stddev(point, camera_positions):
    """
    Return the standard deviation of the distances between point and the
    position of the camera's optical center for each photo.
    """
    return point_to_positions(point, camera_positions).std()
    

def fixed_point_distances(reference_vector, camera_positions, camera_orientations,
                          verbose=False):
    """
    Calculate the distances between the estimates of the fixed point location
    for each photo and the centroid of all the fixed point estimates.
    """
    # Calculate and return distances
    estimates = []
    distances = zeros(len(camera_positions))
    for i, orientation in enumerate(camera_orientations):
        camera_rotation_matrix = cv2.Rodrigues(orientation)[0]
        camera_direction = camera_rotation_matrix.dot(array([[0], [0], [1]]))
        # camera coordinate system
        ccz = row_vector(camera_direction)
        ccx_vector = cross([0, 1, 0], ccz)
        ccx = ccx_vector/norm(ccx_vector)
        ccy = cross(ccz, ccx)
        estimate = (camera_positions[i] +
                    reference_vector[0]*column_vector(ccx) +
                    reference_vector[1]*column_vector(ccy) +
                    reference_vector[2]*column_vector(ccz))
        estimates.append(estimate)
    centroid = sum(array(estimates)) / len(estimates)
    distances = array([norm(e - centroid) for e in estimates])
    if verbose:
        print "centroid:", row_vector(centroid)
        print "distances:", distances
        print "mean distance:", distances.mean()
        print "standard deviation of distances:", distances.std()
        print "reference vector:", reference_vector
        plot_vectors_from_center(centroid, camera_positions)
        plot_vectors(camera_positions, estimates)
        plot_points(camera_positions)
    return distances
    

def fixed_point_distances_sum(x, camera_positions, camera_orientations,
                              verbose=False):
    """
    Return the sum of the distances between a fixed point in space and the
    position of the camera's optical center for each photo.
    """
    return fixed_point_distances(x, camera_positions, camera_orientations).sum()
                                     

def fixed_point_distances_std(x, camera_positions, camera_orientations,
                              verbose=False):
    """
    Return the standard deviation of the distances between a fixed point in
    space and the position of the camera's optical center for each photo.
    """
    return fixed_point_distances(x, camera_positions, camera_orientations).std()
                                     

def print_camera_parameters(reprojection_error, camera_matrix,
                            distortion_coefficients, reference_vector,
                            distances):
    """
    Print the camera's instrinsic parameter matrix and distortion coefficients in
    python format so it can be redirected to a file.
    """
    print
    print "# Do not edit this file, it is auto-generated using",
    print "camera_calibration.py."
    print
    print "# Reprojection error in pixels, a measure of calibration quality"
    print "#", reprojection_error
    print
    print "# Import statements"
    print "from numpy import array, float32"
    print
    print '"""'
    print "Camera properties from the manufacturer"
    print '"""'
    print "pixel_width = %d" % CAMERA_RESOLUTION[0]
    print "pixel_height = %d" % CAMERA_RESOLUTION[1]
    print
    print
    print '"""'
    print "Camera properties found using OpenCV"
    print '"""'
    print
    print "# These are the radial and tangential distortion coefficients."
    print
    dc_string = ("distortion_coefficients = array([[ %10f, %10f, " +
                 "%10f, %10f, %10f ]])")
    print dc_string % tuple(f for f in distortion_coefficients[0])
    print
    print
    """
    Transform camera_matrix to match the coordinate system used in
    dome_projection.py and dome_calibration.py.  This rotates the camera so it
    points along the y-axis instead of the z-axis.
    """
    print "# This matrix contains the intrinsic camera properties."
    print
    cm_string = ("matrix = array([[ %10.3f, %10.3f, %10.3f ],\n" +
                 "                [ %10.3f, %10.3f, %10.3f ],\n" +
                 "                [ %10.3f, %10.3f, %10.3f ]], dtype=float32)")
    print cm_string % tuple(f for row in camera_matrix for f in row)
    print
    print
    print "# The mean distance from the centroid of reference point estimates"
    print "# to each estimate for 9 photos."
    print "#", distances.mean()
    print
    print "# The standard deviation of distances from the centroid of"
    print "# reference point estimates to each estimate for 9 photos."
    print "#", distances.std()
    print
    print "# Vector in meters from the camera's optical center to a"
    print "# stationary reference point that is independent of camera"
    print "# orientation.  This vector is in terms of the camera's"
    print "# coordinate system."
    print
    ro_string = ("reference_vector = array([%10f, %10f, %10f])")
    print ro_string % tuple(f for f in reference_vector)


if __name__ == "__main__":
    """
    Use all files in the moving_tripod and moving_camera_remotely directories
    with the extension .jpg for camera calibration.  In each picture look for a
    16 by 9 checker board pattern and find the pixel coordinates of its 15 by 8
    pattern of "internal" corners. 
    """
    filenames = glob.glob('calibration/foscam_FI9821P/moving_tripod/*.jpg')
    filenames.extend(glob.glob('calibration/foscam_FI9821P/moving_camera_remotely/*.jpg'))
    image_points = find_image_points(filenames)
    object_points = [] # 3d points in real world space
    for i in range(len(image_points)):
        object_points.append(checker_board)
    
    """
    Use the coordinates of the images "internal" corners found above to find
    the camera's instrinsic parameter matrix and its distortion coefficients.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)
    (reprojection_error, camera_matrix,
    distortion_coefficients, rotation_vectors, translation_vectors) = \
            cv2.calibrateCamera(object_points, image_points,
                                CAMERA_RESOLUTION, criteria=criteria)
    
    if DEBUG:
        aperture_width = 0
        aperture_height = 0
        print "Calibration matrix values output"
        returns = cv2.calibrationMatrixValues(camera_matrix, camera_resolution,
                                      aperture_width, aperture_height)
        fovx, fovy, focalLength, principalPoint, aspectRatio = returns
        print "Horizontal field of view:", fovx
        print "Vertical field of view:", fovy
        print "Focal length:", focalLength
        print "Principal point:", principalPoint
        print "Aspect ratio:", aspectRatio

    
    """
    Use all files in the moving_camera_remotely directory with the extension
    .jpg to find a stationary reference point that's location is independent
    of camera orientation and is the same distance from the optical center
    regardless of orientation.
    """
    filenames = glob.glob('calibration/foscam_FI9821P/moving_camera_remotely/*.jpg')
    object_positions, object_orientations = \
            find_object_positions_and_orientations(filenames)
    camera_positions, camera_orientations = \
            find_camera_positions_and_orientations(camera_matrix,
                                                   object_positions,
                                                   object_orientations)
    """
    Calculate a vector which represents the distance and direction from the
    camera's optical center to the fixed point in space given the camera's
    orientation.
    """
    offset_vector = [0, 0, 0]
    arguments = (camera_positions, camera_orientations)
    results = minimize(fixed_point_distances_sum, offset_vector,
                       args=arguments, method='Nelder-Mead')
    reference_vector = results['x']
    distances = fixed_point_distances(results['x'], camera_positions,
                                      camera_orientations)

    if DEBUG:
        print results 
        fixed_point_distances(results['x'], camera_positions,
                              camera_orientations, verbose=True)
    
    """
    Print the camera's instrinsic parameter matrix and distortion coefficients
    in python format so it can be redirected to a file.
    """
    print_camera_parameters(reprojection_error, camera_matrix,
                            distortion_coefficients, reference_vector,
                            distances)
    
