from numpy import array, zeros, uint8, floor
from numpy.linalg import norm
from random import randint
from PIL import Image
import cv2

# import from a relative directory
import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import find_center_points
import foscam_FI9821P as camera

def neighbors(pixel):
    """ return a list of the pixels that share a border with this one """
    x, y = pixel
    #neighbors = [[x - 1, y + 1], [x, y + 1], [x + 1, y + 1],
    #             [x - 1, y],                 [x + 1, y],
    #             [x - 1, y - 1], [x, y - 1], [x + 1, y - 1]]
    neighbors = [                [x, y + 1],
                 [x - 1, y],                 [x + 1, y],
                                 [x, y - 1]]
    return neighbors


class PixelObject():
    """ Define a class to enforce the constraint that the object consist of
    contiguous image pixels.  Instead of defining each pixel independently it defines a
    starting pixel and the path from it to draw the object. """
    def __init__(first_pixel):
        """ initialize the list of pixels that the object contains and the list
        of pixels that are eligible to be added to the object """
        self.pixels = [first_pixel]
        x, y = first_pixel
        self.eligible = neighbors(first_pixel)


    def centroid(self):
        """ return the arithmetic mean of the rows and columns of all the
        pixels in the object """
        x = mean([p[0] for p in self.pixels])
        y = mean([p[1] for p in self.pixels])
        return x, y


    def add_pixel(self, pixel):
        """ add an eligible pixel to the object """
        if pixel in self.eligible:
            # add this pixel to the object
            self.pixels.append(pixel)
            # remove this pixel from the eligible list
            self.eligible.pop(pixel)
            # add the neighbors of this pixel to the eligible list if they are
            # not yet on it
            for neighbor in neighbors(pixel):
                if neighbor not in self.eligible:
                    self.eligible.append(neighbor)


def sort_points(points, num_columns, num_rows):
    """
    Take a set of points that are known to be arranged in a grid layout and
    that are already sorted by their y values and sort them from left to right
    and top to bottom.
    """
    points = list(points)
    rows = []
    for r in range(num_rows):
        row = points[r * num_columns : (r + 1) * num_columns]
        rows.append(row)
        # sort each row of points by x value
        rows[-1].sort(key=lambda p:p[0])

    sorted_points = []
    for row in rows:
        sorted_points.extend(row)

    return sorted_points


def compare_centroids(expected, actual, tolerance):
    """ test two sets of centroids for equivalence within a given tolerance """
    assert len(expected) == len(actual), \
            "Wrong number of centroids found in image!"
    for i in range(len(expected)):
        assert norm(array(expected[i]) - array(actual[i])) < tolerance
        if norm(array(expected[i]) - array(actual[i])) > tolerance:
            print "Centroid found is not within the required tolerance!"
            print "Expected centroid:", expected[i]
            print "Centroid found:", actual[i]


def gen_test_image(columns, rows):
    """ create an image with a grid of objects centered on known pixel values """
    dx = camera.pixel_width/columns
    dy = camera.pixel_height/rows
    # creat a list of top left pixels that will be used to draw the objects
    top_left_pixels = []
    # arrange the objects in a grid with some random variation
    for row in range(dy/2, camera.pixel_height - 1, dy):
        for column in range(dx/2, camera.pixel_width - 1, dx):
            rand_dx = randint(-10, 10)
            rand_dy = randint(-10, 10)
            top_left_pixels.append([row + rand_dy, column + rand_dx])
    # calculate centroids for use as expected result from find_center_points
    centroids = [[p[1] + 1.0, p[0] + 1.0] for p in top_left_pixels]

    # draw a square of four pixels for each object
    pixels = zeros([camera.pixel_height, camera.pixel_width], dtype=uint8)
    for i in range(len(top_left_pixels)):
        pixel = top_left_pixels[i]
        left = int(pixel[1])
        right = left + 1
        top = int(pixel[0])
        bottom = top + 1
        for row in [top, bottom]:
            for col in [left, right]:
                pixels[row, col] = 255

    test_image = Image.fromarray(pixels, mode='L')
    #test_image.show()
    return test_image, centroids


def gen_image_from_corners(corners):
    """ make an image containing objects with centroids at the corner locations
    by centering circles on these locations """
    pixels = zeros([camera.pixel_height, camera.pixel_width], dtype=uint8)
    radius = 11
    for corner in corners:
        # convert (u, v) corner coordinates to centroid pixel coordinates
        centroid = corner[0] - array([0.5, 0.5])
        centroid_x = int(round(centroid[0]))
        centroid_y = int(round(centroid[1]))
        #object_pixels = []
        for y in range(centroid_y - radius, centroid_y + radius + 1):
            for x in range(centroid_x - radius, centroid_x + radius + 1):
                point = array([x, y])
                if norm(point - centroid) < radius:
                    #object_pixels.append(point)
                    pixels[y, x] = 255
        #print sum(object_pixels)/float(len(object_pixels)) - centroid
    test_image = Image.fromarray(pixels, mode='L')
    #test_image.show()
    return test_image


def fmin_int(x, xn0, xd):
    """ minimize x - sum(xn/xd) """
    #for each numerator in xn try increasing it and see if the difference
    # changes
    xn = array(xn0)
    smallest_difference = abs(x - sum(xn*xd)/float(sum(xn)))
    for j in range(100):
        for i in range(len(xn)):
            xn_try = array(xn)
            xn_try[i] = randint(1, 10)
            print "xn_try", xn_try
            #import pdb; pdb.set_trace()
            try_diff = abs(x - sum(xn_try*xd)/float(sum(xn_try)))
            if try_diff < smallest_difference:
                xn = xn_try
                smallest_difference = try_diff
    return xn, smallest_difference


def gen_object_from_centroid(centroid, width=2, tolerance=1e-1):
    """ Assume a solution of the form (for width=2):
        x = (a1*floor(x) + a2*(floor(x) + 1))/(a1 + a2)
        y = (b1*floor(y) + b2*(floor(y) + 1))/(b1 + b2)
        a1 + a2 = b1 + b2
    """
    x, y = centroid[0]
    xd = array(range(width)) - (width - 1)/2 + floor(x)
    yd = array(range(width)) - (width - 1)/2 + floor(y)
    # for now just try to find the numerators that 
    # minimize x - sum(xn/xd)
    xn = array(xd == floor(x), dtype='int')
    xn = fmin_int()



    # minimize norm((x, y) - (sum(xn/xd), sum(yn/yd))


# this is a constrained optimization problem the underlying constraint is that
# the pixels form a continuous object


    # find the denominators for the fractions
    #x = floor(centroid[0][0])
    #y = floor(centroid[0][1])
    # find the numerators for the fractions


def gen_image_from_corners2(corners):
    # make objects with centroids at the corner locations by centering circles on
    # these locations
    pixels = zeros([camera.pixel_height, camera.pixel_width], dtype=uint8)
    radius = 11
    for center in corners:
        center_x = int(round(center[0, 0]))
        center_y = int(round(center[0, 1]))
        object_pixels = []
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                point = array([x, y])
                if norm(point - center) < radius:
                    object_pixels.append(point)
                    pixels[y, x] = 255
        #print sum(object_pixels)/float(len(object_pixels)) - center[0]
    test_image = Image.fromarray(pixels, mode='L')
    #test_image.show()
    return test_image


def corners_to_points(corners):
    """ convert a list of corners returned by cv2.findChessboardCorners into an
    array of (x, y) pairs """
    #for i in range(len(corners)):
        # flip x and y to match row, column format of find_center_points
        #target_center_point = array([corners[i, 0, 1], corners[i, 0, 0]])
    return array([[c[0, 0], c[0, 1]] for c in corners])


def test_find_center_points_on_simple_test_image():
    """
    Use gen_test_image to make an image containing objects with known centroids.
    Then use find_center_points to find the centroids of the objects and confirm
    that they are the same.
    """
    required_accuracy = 1e-6
    columns = 16
    rows = 9
    test_image, expected_centroids = gen_test_image(columns, rows)
    actual_centroids = find_center_points(test_image, blur=0, remove_distortion=False)
    actual_centroids = sort_points(actual_centroids, columns, rows)
    compare_centroids(expected_centroids, actual_centroids, required_accuracy)
    
    
def off_test_find_center_points():
    """
    This time use cv2.findChessboardCorners on a real photograph and use the
    corners it returns to generate an image containing objects with those corners
    as centroids.  Then use find_center_points with remove_distortion=False on this
    image and compare its results to OpenCV's.
    """
    DEBUG = False
    
    # load the chess board image
    camera_resolution = (camera.pixel_width, camera.pixel_height)  # (x, y)
    image = cv2.imread("tripod_closeup.jpg")
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert gray_scale_image.shape[::-1] == camera_resolution
    
    # Invert the gray scale image because findChessboardCorners requires a
    # white background.
    inverted_image = 255 - gray_scale_image
    
    # Find the chess board corners
    BOARD_SIZE = (15, 8)  # OpenCV board size, basically (cols - 1, rows - 1)
    found, corners = cv2.findChessboardCorners(inverted_image,
                                               BOARD_SIZE,
                                               flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                               cv2.CALIB_CB_ASYMMETRIC_GRID)
    expected_centroids = corners_to_points(corners)
        
    if DEBUG:
        # Show each image with the corners drawn on
        cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
        cv2.imshow('Corners', image)
    
    # Generate an image using the corners to place the objects and use
    # find_center_points to see if we get the (u, v) coordinates of corners back
    test_image = gen_image_from_corners(corners)
    center_points = find_center_points(test_image, remove_distortion=False)
    actual_centroids = sort_points(center_points, 15, 8)
    required_accuracy = 0.2
    #required_accuracy = 0.1
    compare_centroids(expected_centroids, actual_centroids, required_accuracy)
    
    print "Done with photo find_center_points test"
    
    
    #sys.exit()
    
    
def off_test_find_center_points2():
    """
    This time use cv2.undistort to remove the distortion from the photograph and
    then compare its results to find_center_points with remove_distortion=True.
    """
    DEBUG = False
    # undistort the inverted_image from above and find the corners again
    
    undist_image = cv2.undistort(inverted_image, camera.matrix,
                                 camera.distortion_coefficients)
    
    # Find the chess board corners
    found, corners = cv2.findChessboardCorners(undist_image,
                                               BOARD_SIZE,
                                               flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                               cv2.CALIB_CB_ASYMMETRIC_GRID)
    expected_centroids = corners_to_points(corners)
        
    if DEBUG:
        # Show each image with the corners drawn on
        cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)
        cv2.imshow('Corners', image)
        #cv2.waitKey(500)
    
    # find the center points and remove distortion
    center_points = find_center_points(test_image, remove_distortion=True)
    actual_centroids = sort_points(center_points, 15, 8)
    
    # compare the points found in the image with corners from the chessboard
    required_accuracy = 0.7
    compare_centroids(expected_centroids, actual_centroids, required_accuracy)
    
    #for i in range(len(corners)):
        # flip x and y to match row, column format of find_center_points
        #target_center_pixel = array([corners[i, 0, 1], corners[i, 0, 0]])
        #center_pixel = array(center_points[i])
        #if norm(target_center_pixel - center_pixel) > required_accuracy:
            #print "Unexpected pixel found!"
            #print "Expected center pixel:", target_center_pixel
            #print "Center pixel found:", center_pixel
    print "Done with distortion removal find_center_points test"
    
    
    
    
    """
    test = np.zeros((10,1,2), dtype=np.float32)
    xy_undistorted = cv2.undistortPoints(test, camera_matrix, dist_coeffs)
    """
    
