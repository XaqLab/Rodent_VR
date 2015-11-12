from numpy import array, zeros, uint8
from PIL import Image

PROJECTOR_PIXEL_WIDTH = 1280
PROJECTOR_PIXEL_HEIGHT = 720
GREEN = array([0, 128, 0], dtype=uint8)

centroids = zeros([22,2])
# Read the parameters from a text file
file_name = "calibration/calibration_device/2015_11_06_centroids.txt"
if file_name:
    try:
        parameter_file = open(file_name, "r")
        for i in range(len(centroids)):
            line = parameter_file.readline()
            row, col = line.split(", ")
            centroids[i] = [float(row), float(col)]
        parameter_file.close()
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
image.save("caldev_image.png", "png")

