#!/usr/local/bin/python
# written for python 2.7.8

from PIL import Image
from dome_projection import DomeProjection
import sys
import re

DEBUG = False

# Choose a VR system, rat or mouse.
#system = "rat"
system = "mouse"

# Choose the number of images and corresponding yaw angles for each.
num_images = 1; yaw = [0]             # one image in front
#num_images = 3; yaw = [-90, 0, 90]   # three images left, center, right

if system == "rat":
    # Use parameters for the rat VR system.
    parameters = lambda image_size, num_images, yaw:{
        "animal_position": [0.0, -0.139974099238, 0.546942519452],
        "dome_center": [0.0, 0.0148932688144, 0.46755637477],
        "dome_radius": 0.541409227296,
        "mirror_radius": 0.222935706293,
        "projector_focal_point": [0.0, 0.84950020408, -0.00296533955158],
        "projector_roll": -0.0209584381456,
        "projector_theta": 0.269020062357,
        "projector_vertical_offset": 0.183968098375,
        "screen_height": [image_size[1]]*num_images,
        "screen_width": [image_size[0]]*num_images,
        "distance_to_screen": [0.3*image_size[1]]*num_images,
        "pitch": [10]*num_images,
        "yaw": yaw,
        "image_pixel_width": [image_size[0]]*3,
        "image_pixel_height": [image_size[1]]*3,
        "projector_pixel_width": 1920,
        "projector_pixel_height": 1080
    }
elif system == "mouse":
    # Use parameters for the mouse VR system.
    parameters = lambda image_size, num_images, yaw:{
        "animal_position": [0, 0.056, 0.575],
        "dome_center": [0, 0.110, 0.348],
        "dome_radius": 0.601,
        "mirror_radius": 0.162,
        "projector_focal_point": [0, 0.858, -0.052],
        "projector_roll": 0.01,
        "projector_theta": 0.166,
        "projector_vertical_offset": 0.211,
        "screen_height": [image_size[1]]*num_images,
        "screen_width": [image_size[0]]*num_images,
        "distance_to_screen": [2.5*image_size[1]]*num_images,
        "pitch": [10]*num_images,
        "yaw": yaw,
        "image_pixel_width": [image_size[0]]*3,
        "image_pixel_height": [image_size[1]]*3,
        "projector_pixel_width": 1280,
        "projector_pixel_height": 720
    }

dome_image_size = (0, 0)
# Warp each file passed as an argument for projection on the dome
for filepath in sys.argv[1:]:
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
        print "Creating instance of DomeProjection Class"
        dome = DomeProjection(**parameters(dome_image_size, num_images, yaw))
        print "Done initializing dome"
    
    output_image = dome.warp_image_for_dome([input_image]*3)
    
    # save warped image
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
    
