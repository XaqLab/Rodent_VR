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
Prepare points for the checker board's "internal" corners.  The checker board has
9 rows and 16 columns, this results in 15 "internal" corners per row and 8 per
column.  The measured viewing area on the TV I used for calibration was 882 mm
wide by 488 mm high.  
"""
dx = 882.0/16
dy = 488.0/9
checker_board = array([[x*dx, y*dy, 0]
                       for y in range(BOARD_SIZE[1])
                       for x in range(BOARD_SIZE[0])], float32)

"""
Use all files in the moving_tripod directory with the extension .jpg for
camera calibration.  In each picture look for a 16 by 9 checker board pattern
and find the pixel coordinates of its 15 by 8 pattern of "internal" corners. 
"""
filenames = glob.glob('calibration/foscam_FI9821P/moving_tripod/*.jpg')
files_used = []    # files in which the checker board was found
object_points = [] # 3d points in real world space
image_points = []  # 2d points in image plane.
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
        print "Found checker board in", filename
        files_used.append(filename)
        object_points.append(checker_board)
        image_points.append(corners)

        # Draw and display the corners
        if DEBUG:
            cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
            cv2.imshow('Corners', image)
            cv2.waitKey(500)
    else:
        print "No checker board in", filename

if DEBUG:
    cv2.destroyAllWindows()

"""
Use the coordinates of the images "internal" corners found above to find the
camera's instrinsic parameter matrix and its distortion coefficients.
"""
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

    for i in range(len(files_used)):
        print
        print files_used[i]
        print "Rotation:"
        print rotation_vectors[i]
        print 180/pi*norm(rotation_vectors[i])
        print "Translation:"
        print translation_vectors[i]


"""
Use all files in the moving_camera_remotely directory with the extension .jpg
to find a stationary reference point that's location is independent of camera
orientation.
"""
print
filenames = glob.glob('moving_camera_remotely/*.jpg')
files_used = []    # files in which the checker board was found
rotation_vectors = []
translation_vectors = []
for filename in filenames:
    image = cv2.imread(filename)
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the gray scale image because findChessboardCorners requires a
    # white background.
    inverted_image = 255 - gray_scale_image
    # Remove radial distortion from the image
    undist_image = cv2.undistort(inverted_image, camera_matrix,
                                distortion_coefficients)
    #undist_image = inverted_image

    # Find the chess board corners
    found, corners = cv2.findChessboardCorners(undist_image,
                                               BOARD_SIZE,
                                               flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                               cv2.CALIB_CB_ASYMMETRIC_GRID)

    # If found, add object points, image points
    if found == True:
        files_used.append(filename)
        outputs = cv2.solvePnP(checker_board, corners,
                               camera_matrix, distortion_coefficients)
        retval, rotation_vector, translation_vector = outputs
        rotation_vectors.append(rotation_vector)
        translation_vectors.append(translation_vector)

        # Draw and display the corners
        if True:
            cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
            cv2.imshow('Corners', image)
            cv2.waitKey(500)
    else:
        print "Checker board not found in", filename

if True:
    cv2.destroyAllWindows()

for i in range(len(files_used)):
    print
    print files_used[i]
    print "Rotation:"
    print rotation_vectors[i]
    print 180/pi*norm(rotation_vectors[i])
    print "Translation:"
    print translation_vectors[i]


"""
Transform camera_matrix to match the coordinate system used in
dome_projection.py and dome_calibration.py.  This rotates the camera so it
points along the y-axis instead of the z-axis.
"""
camera_rotation_vector = array([pi/2, 0, 0])
camera_rotation_matrix = cv2.Rodrigues(camera_rotation_vector)[0]
rotated_camera_matrix = camera_matrix.dot(camera_rotation_matrix)

"""
Print the camera's instrinsic parameter matrix and distortion coefficients in
python format so it can be redirected to a file.
"""
print "# Do not edit this file, it is auto-generated using",
print "camera_calibration.py."
print
print "# Reprojection error in pixels, a measure of calibration quality"
print reprojection_error
print
print "# Import statements"
print "from numpy import array, float32"
print
print '"""'
print "Camera properties from the manufacturer"
print '"""'
print "pixel_width = %d" % camera_resolution[0]
print "pixel_height = %d" % camera_resolution[1]
print
print '"""'
print "Camera properties found using OpenCV"
print '"""'
print
print "# These are the radial and tangential distortion coefficients."
dc_string = ("distortion_coefficients = array([[ %10f, %10f, " +
             "%10f, %10f, %10f ]])")
print dc_string % tuple(f for f in distortion_coefficients[0])
print
print "# This matrix contains the intrinsic camera properties."
cm_string = ("matrix = array([[ %10.3f, %10.3f, %10.3f ],\n" +
             "                [ %10.3f, %10.3f, %10.3f ],\n" +
             "                [ %10.3f, %10.3f, %10.3f ]], dtype=float32)")
print cm_string % tuple(f for row in rotated_camera_matrix for f in row)
print
print "# Direction and distance from optical center to a stationary reference"
print "# point that is independent of camera orientation."
print
#rr_string = ("reference_rotation = array([[ %10f, %10f, %10f ]])")
#print rr_string % tuple(f for f in reference_rotation[0])
#print
#rd_string = ("reference_distance = %10f")
#print rd_string % reference_distance
print
print '"""'
print "Measured camera properties"
print '"""'
print
print "# This is the distance, in meters, from the intersection of the"
print "# pitch and yaw rotation axes to the center of the lens.  This value"
print "# was measured with a tape measure."
print "axes_to_lens = 0.035"

