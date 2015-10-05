#!/usr/local/bin/python
# written for python 2.7.8

from PIL import Image
from dome_projection import DomeProjection
import sys
import re

DEBUG = False

dome_image_size = (0, 0)
# Warp each file passed as an argument for projection on the dome
for filepath in sys.argv[1:]:
    filename = re.split(r"/",filepath)[-1]
    if "jpg" in filename.lower():
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
                              #screen_height = [2.375],
                              #screen_width = [4.0625],
                              #distance_to_screen = [3.0],
        dome = DomeProjection(
                              screen_height = [dome_image_size[1]],
                              screen_width = [dome_image_size[0]],
                              distance_to_screen = [2.5*dome_image_size[1]],
                              pitch = [10],
                              yaw = [0],
                              image_pixel_width=[dome_image_size[0]],
                              image_pixel_height=[dome_image_size[1]],
                              projector_pixel_width=1280,
                              projector_pixel_height=720)
        print "Done initializing dome"
    
    output_image = dome.warp_image_for_dome([input_image])
    
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
    
