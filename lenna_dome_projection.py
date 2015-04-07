#!/usr/local/bin/python
# written for python 2.7.8

############################################################
# Main Program Starts Here
############################################################
from PIL import Image
from dome_projection import DomeProjection

directory = "test_images/512_by_512/"
filename = "lenna.png"
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
dome = DomeProjection(screen_height = [1.0],
                      screen_width = [1.0],
                      distance_to_screen = [0.5],
                      pitch = [30],
                      yaw = [0],
                      roll = [0],
                      image_pixel_width=[512],
                      image_pixel_height=[512],
                      projector_pixel_width=1280,
                      projector_pixel_height=720)
print "Done initializing dome"

output_image = dome.warp_image_for_dome([input_image])

output_image.show()
output_image.save(directory + "warped_" + filename, "PNG")

unwarped_images = dome._unwarp_image(output_image)
unwarped_images[0].show()
unwarped_images[0].save(directory + "unwarped_" + filename, "PNG")

