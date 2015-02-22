#!/usr/local/bin/python
# written for python 2.7.8

from PIL import Image
from dome_projection import DomeProjection
from os import listdir


directory = "three_camera_demo/"
filenames = listdir(directory + "center")

print "Creating instance of DomeProjection Class"
dome = DomeProjection()
print "Done initializing dome"
        
for filename in filenames:
    if "stereo" in filename:
        print "Opening image in", filename
        left_image = Image.open(directory + "left/" + filename)
        center_image = Image.open(directory + "center/" + filename)
        right_image = Image.open(directory + "right/" + filename)

        output_image = dome.warp_image_for_dome([left_image, center_image,
                                                 right_image])
        output_filename = "warped_" + filename
        print "Saving warped image to:", output_filename
        output_image.save(directory + output_filename, "jpeg")
        
        
