"""
This is the script I wrote to characterize the camera used to estimate the dome
projection geometry.  I displayed a black and white checkerboard pattern
16 squares wide and 9 squares high, matching the TV's 16:9 aspect ratio.
I took 10 pictures of this pattern with the camera, moving the camera between
pictures.
"""
from numpy import array, zeros, float32, mgrid, pi, arctan
from numpy.linalg import norm
import cv2  # open computer vision library
import glob
from PIL import Image


BOARD_SIZE = (15, 8)  # (x, y)
DEBUG = False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

"""
Prepare points for the checker board's "inside" corners.  The checker board has
9 rows and 16 columns, this results in 15 "inside" corners per row and 8 per
column.  The measured viewing area on the TV I used for calibration was 882 mm
wide by 488 mm high.  
"""
dx = 882.0/16
dy = 488.0/9
checker_board = array([[x*dx, y*dy, 0]
                       for y in range(BOARD_SIZE[1])
                       for x in range(BOARD_SIZE[0])], float32)

# Arrays to store object points and image points from all the images.
object_points = [] # 3d points in real world space
image_points = []  # 2d points in image plane.

filenames = glob.glob('*.jpg')

camera_resolution = None

for filename in filenames:
    image = cv2.imread(filename)
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the gray scale image because findChessboardCorners requires a
    # white background.
    inverted_image = 255 - gray_scale_image
    if camera_resolution:
        assert inverted_image.shape[::-1] == camera_resolution
    else:
        camera_resolution = inverted_image.shape[::-1]

    # Find the chess board corners
    found, corners = cv2.findChessboardCorners(inverted_image,
                                               BOARD_SIZE, None)

    # If found, add object points, image points
    if found == True:
        object_points.append(checker_board)
        image_points.append(corners)

        # Draw and display the corners
        if DEBUG:
            cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
            cv2.imshow('Corners', image)
            cv2.waitKey(500)

if DEBUG:
    cv2.destroyAllWindows()


# find the camera parameters using the object points and image points
outputs = cv2.calibrateCamera(object_points, image_points,
                              camera_resolution, criteria=criteria)
(reprojection_error, camera_matrix, distortion_coefficients, rotation_vectors,
 translation_vectors) = outputs

if DEBUG:
    filename = filenames[1]
    dist_image = cv2.imread(filename)
    corners = image_points[1]
    cv2.drawChessboardCorners(dist_image, BOARD_SIZE, corners, True)
    cv2.imwrite("junk/dist_image.jpg", dist_image)
    dist_image = cv2.imread(filename)
    undist_image = cv2.undistort(dist_image, camera_matrix,
                                distortion_coefficients)
    undist_corners = cv2.undistortPoints(corners, camera_matrix,
                                        distortion_coefficients, R=None,
                                        P=camera_matrix)
    cv2.drawChessboardCorners(undist_image, BOARD_SIZE, undist_corners, True)
    cv2.imwrite("junk/undist_image.jpg", undist_image)

# x and y focal lengths in pixels
fpx = camera_matrix[0, 0]
fpy = camera_matrix[1, 1]

# x and y coordinates of the principal point in the image plane
ppx = camera_matrix[0, 2]
ppy = camera_matrix[1, 2]

# x and y fields of view
left_field_of_view = 180/pi*arctan((0 - ppx)/fpx)
right_field_of_view = 180/pi*arctan((camera_resolution[0] - ppx)/fpx)
upper_field_of_view = -180/pi*arctan((0 - ppy)/fpy)
lower_field_of_view = -180/pi*arctan((camera_resolution[1] - ppy)/fpy)

if DEBUG:
    print "Reprojection error"
    print reprojection_error
    print
    print "Distortion coefficients"
    print distortion_coefficients
    print
    print "Camera matrix"
    print camera_matrix
    print
    # fovx, fovy, focalLength, principalPoint, aspectRatio
    aperture_width = 0
    aperture_height = 0
    print "Calibration matrix values output"
    print cv2.calibrationMatrixValues(camera_matrix, camera_resolution,
                                  aperture_width, aperture_height)
    print
    print "Fields of view"
    print "left:", left_field_of_view
    print "right:", right_field_of_view
    print "horizontal:", right_field_of_view - left_field_of_view
    print "upper:", upper_field_of_view
    print "lower:", lower_field_of_view
    print "vertical:", upper_field_of_view - lower_field_of_view
    print

    print len(filenames)
    print len(rotation_vectors)
    print len(translation_vectors)
    for i in range(len(filenames)):
        print
        print filenames[i]
        print "Rotation:"
        print rotation_vectors[i]
        print 180/pi*norm(rotation_vectors[i])
        print "Translation:"
        print translation_vectors[i]

print "# Import statements"
print "from numpy import array, float32"
print
print "# Camera properties found using OpenCV"
print "pixel_width = %d" % camera_resolution[0]
print "pixel_height = %d" % camera_resolution[1]
print "fpx = %f" % fpx
print "fpy = %f" % fpy
print "ppx = %f" % ppx
print "ppy = %f" % ppy
dc_string = ("distortion_coefficients = array([[ %8f, %8f, " +
             "%8f, %8f, %8f ]])\n")
print dc_string % tuple(c for c in distortion_coefficients[0])
cm_string = ("matrix = array([[ %8.3f, %8.3f, %8.3f ],\n" +
             "                [ %8.3f, %8.3f, %8.3f ],\n" +
             "                [ %8.3f, %8.3f, %8.3f ]], dtype=float32)\n")
print cm_string % tuple(f for row in camera_matrix for f in row)

print "# Camera properties found NOT using OpenCV"
print
print "# This is the distance, in meters, from the intersection of the"
print "# pitch and yaw rotation axes to the center of the lens.  This value"
print "# was measured crudely with a tape measure."
print "axes_to_lens = 0.035"
print
print "# This is the focal distance, in meters, of the camera's lens.  I just"
print "# guessed a value."
print "focal_length = 0.001"

