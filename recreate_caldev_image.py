from numpy import array, zeros, uint8
from PIL import Image
import sys

"""
Create an image with green dots in the projector's image that correspond to the
viewing directions from the calibration device. After calibration these green
dots should be very close to the center of the spots from the calibration
device.
"""

#PROJECTOR_PIXEL_WIDTH = 1280
#PROJECTOR_PIXEL_HEIGHT = 720
PROJECTOR_PIXEL_WIDTH = 1024
PROJECTOR_PIXEL_HEIGHT = 768
GREEN = array([0, 255, 0], dtype=uint8)

centroids = zeros([22,2])
# Read the centroids from a text file
file_name = sys.argv[1]
if file_name:
    try:
        centroid_file = open(file_name, "r")
        for i in range(len(centroids)):
            line = centroid_file.readline()
            row, col = line.split(", ")
            centroids[i] = [float(row), float(col)]
        centroid_file.close()
    except Exception, e:
        print str(e)
        raise e


pixels = zeros([PROJECTOR_PIXEL_HEIGHT, PROJECTOR_PIXEL_WIDTH, 3], dtype=uint8)
for centroid in centroids:
    row, col = centroid
    row = int(row)
    col = int(col)
    if (row >= 0 and row < PROJECTOR_PIXEL_HEIGHT and
        col >= 0 and col < PROJECTOR_PIXEL_WIDTH):
        pixels[row, col] = GREEN
        pixels[row, col + 1] = GREEN
        pixels[row + 1, col] = GREEN
        pixels[row + 1, col + 1] = GREEN

image = Image.fromarray(array(pixels, dtype=uint8), mode='RGB')
image.save("recreated_caldev_image.png", "png")

