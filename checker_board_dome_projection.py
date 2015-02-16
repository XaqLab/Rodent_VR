#!/usr/local/bin/python
# written for python 2.7.8

############################################################
# Main Program Starts Here
############################################################
from PIL import Image
from dome_projection import DomeProjection

directory = "test_images/1280_by_720/"
filename = "checker_board_20_by_20.png"
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

# show original image
input_image.show()

print "Creating instance of DomeProjection Class"
dome = DomeProjection(screen_height = 0.9,
                      screen_width = 1.6,
                      distance_to_screen = 0.5,
                      image_pixel_width=1280,
                      image_pixel_height=720,
                      projector_pixel_width=1280,
                      projector_pixel_height=720)
print "Done initializing dome"

output_image = dome.warp_image_for_dome(input_image)

output_image.show()
output_image.save(directory + "warped_" + filename, "PNG")


