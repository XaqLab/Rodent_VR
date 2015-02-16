#!/usr/local/bin/python
# written for python 2.7.8

############################################################
# Main Program Starts Here
############################################################
from PIL import Image
from dome_projection import DomeProjection
from os import listdir


directory = "scenes28/"
filenames = listdir(directory)

print "Creating instance of DomeProjection Class"
dome = DomeProjection(screen_height = 1.0,
                      screen_width = 1.0,
                      distance_to_screen = 0.5,
                      image_pixel_width=1280,
                      image_pixel_height=720,
                      projector_pixel_width=1280,
                      projector_pixel_height=720)
print "Done initializing dome"
        
for filename in filenames:
    if "stereo" in filename:
        print "Opening image in", filename
        input_image = Image.open(directory + filename)

        # read pixel data from image into a list
        if input_image.mode == 'RGB':
            print "RGB image"
            print input_image.size
        elif input_image.mode == 'L':
            print "Grayscale image"
            print input_image.size
        else:
            print "Unsupported image mode:", input_image.mode
            exit()
        
        output_image = dome.warp_image_for_dome(input_image)
        output_filename = "warped_" + filename
        print "Saving warped image to:", output_filename
        output_image.save(directory + output_filename, "jpeg")
        
        
