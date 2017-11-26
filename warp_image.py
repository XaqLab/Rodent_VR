#!/usr/local/bin/python
# written for python 2.7.8

from PIL import Image
from dome_projection import DomeProjection
import sys
import re

if len(sys.argv) != 3:
    # No arguments given.
    print "Usage", sys.argv[0], "parameter_file image_file"


# if DEBUG is True unwarp image to confirm it looks like the original.
DEBUG = False

# Specify the number of images and yaw angles for each.
num_images = 1; yaw = [0]             # one image in front
#num_images = 3; yaw = [-90, 0, 90]   # three images left, center, right

# Read parameters from the file given as the first argument.
parameter_filename = sys.argv[1]
parameter_file = open(parameter_filename)
sys_params = eval(parameter_file.read())

# A function that generates image_params.
def parameters(image_size, num_images, yaw):
    image_params = {
        "screen_height": [image_size[1]]*num_images,
        "screen_width": [image_size[0]]*num_images,
        "distance_to_screen": [0.5*image_size[1]]*num_images,
        "pitch": [30]*num_images,
        "yaw": yaw,
        "image_pixel_width": [image_size[0]]*num_images,
        "image_pixel_height": [image_size[1]]*num_images
    }
    return dict(sys_params.items() + image_params.items())


# Modify image file passed as argument for projection in VR system.
dome_image_size = (0, 0)
filepath = sys.argv[2]
filename = re.split(r"/",filepath)[-1]
if "jpg" in filename.lower() or "jpeg" in filename.lower():
    filetype = "jpeg"
elif "png" in filename.lower():
    filetype = "png"
else:
    exit("unknown file type")
directory = filepath.replace(filename,"")
input_image = Image.open(directory + filename)
if dome_image_size != input_image.size:
    dome_image_size = input_image.size
    # Initialize the dome display
    dome = DomeProjection(**parameters(dome_image_size, num_images, yaw))
print "Building the lookup table and then modifying the image."
print "Building the lookup table will take a few minutes."
output_image = dome.warp_image_for_dome([input_image]*num_images)

# Save modified image.
output_image.save(directory + "warped_" + filename, filetype)

if DEBUG:
    # show original image
    input_image.show()

    # show warped image
    output_image.show()

    # unwarp warped image and show it
    unwarped_images = dome._unwarp_image(output_image)
    unwarped_images[0].show()
    unwarped_images[0].save(directory + "unwarped_" + filename, filetype)

