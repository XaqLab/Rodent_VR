""" Generate the projector calibration image to use with the 3D printed
calibration device """

from numpy import uint8, array, zeros, pi, sin, cos
from PIL import Image
from dome_projection import DomeProjection
import cPickle as pickle

# define the projector resolution
#PROJECTOR_PIXEL_WIDTH = 1280
#PROJECTOR_PIXEL_HEIGHT = 720
PROJECTOR_PIXEL_WIDTH = 1024
PROJECTOR_PIXEL_HEIGHT = 768
GREEN = array([0, 128, 0], dtype=uint8)

# initialize the dome and find the projector pixel coordinates that correspond
# to the calibration directions using the default parameter values
dome = DomeProjection()
calibration_directions = dome.calibration_directions
pickle_filename = 'calibration_image.pkl'
try:
    # see if there are previously found centroids that we can use as an initial
    # guess
    with open(pickle_filename, 'rb') as pickle_file:
        centroids = pickle.load(pickle_file)
    if len(centroids) == len(calibration_directions):
        centroids = dome.find_projector_points(calibration_directions,
                                               centroids)
    else:
        # wrong number of previous centroids so do the search from scratch
        centroids = dome.find_projector_points(calibration_directions)
except IOError:
    # no previous centroids found so do the search from scratch
    centroids = dome.find_projector_points(calibration_directions)

# save centroids to a file for use as the initial guess next time
with open(pickle_filename, 'wb') as pickle_file:
    pickle.dump(centroids, pickle_file)

# make an image with 4 pixel squares centered on these pixel coordinates
pixels = zeros([PROJECTOR_PIXEL_HEIGHT, PROJECTOR_PIXEL_WIDTH, 3], dtype=uint8)

for centroid in centroids:
    # convert centroids from (u, v) coordinates to (row, col) coordinates
    col, row = centroid
    row = int(round(row - 0.5))
    col = int(round(col - 0.5))
    if (row >= 0 and row < PROJECTOR_PIXEL_HEIGHT and
        col >= 0 and col < PROJECTOR_PIXEL_WIDTH):
        pixels[row, col] = GREEN
        pixels[row, col + 1] = GREEN
        pixels[row + 1, col] = GREEN
        pixels[row + 1, col + 1] = GREEN

image = Image.fromarray(array(pixels, dtype=uint8), mode='RGB')
image.save("calibration_image.png", "png")


