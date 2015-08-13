#!/usr/local/bin/python
# written for python 2.7.8

from PIL import Image
from dome_projection import DomeProjection
import sys
import re
import cPickle
import webcam
from numpy import pi

DEBUG = False

filepath = sys.argv[1]
filename = re.split(r"/",filepath)[-1]
if "jpg" in filename.lower():
    filetype = "jpeg"
elif "png" in filename.lower():
    filetype = "png"
else:
    exit("unknown file type")

directory = filepath.replace(filename,"")
input_image = Image.open(directory + filename)

half_theta_yaw = webcam.theta * 180/pi

# screen height and width were calculated based on measurements from webcam photos
# these numbers are in inches
parameters = dict(screen_height = [webcam.screen_height],
                  screen_width = [webcam.screen_width],
                  distance_to_screen = [webcam.distance_to_screen],
                  pitch = [0],
                  yaw = [half_theta_yaw],
                  roll = [0],
                  image_pixel_width = [1280],
                  image_pixel_height = [720],
                  projector_pixel_width = 1280,
                  projector_pixel_height = 720)

print "Creating an instance of the DomeProjection class to warp images"
dome = DomeProjection(**parameters)
#pickle_file = open("dome.cpickle", "rb")
#dome = cPickle.Unpickler(pickle_file).load()
#pickle_file.close()
print "Done initializing dome"
        
print "Opening image in", filename
input_image = Image.open(directory + filename)

print "Warping image"
output_image = dome.warp_image_for_dome([input_image])
output_filename = "warped_" + filename
print "Saving warped image to:", output_filename
output_image.save(directory + output_filename, filetype)


if DEBUG:
    input_image.show()
    print "Creating an instance of the DomeProjection class to unwarp images"
    #parameters['mirror_radius'] -= 0.001
    #parameters['dome_radius'] -= 0.010
    unwarp_dome = DomeProjection(**parameters)
    print "Done initializing dome"
    
    unwarped_images = unwarp_dome._unwarp_image(output_image)
    unwarped_images[0].show()
    unwarped_images[0].save(directory + "unwarped_dome_radius_10mm_too_big_"
                            + filename, filetype)
    
