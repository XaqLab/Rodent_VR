# Rodent_VR

This code modifies (a.k.a. warps) images for display in a virtual reality
system for rodents. The VR system uses a spherical mirror to allow a single
projector to cover a very wide field of view. The projector image is reflected
off of the mirror and on to a hemispherical screen. The projector's image is
distorted by these non-planar surfaces. This code was developed to modify
images for projection in this system such that the original image is seen
inside the VR system. This is achieved by a pixel level mapping which sets the
RGB values for each projector pixel using the RGB values of the pixel in the
original image which is in the same direction from the viewer's point of view.
The viewer in the VR system is the rodent and and the viewers in the virtual
world we created are virtual cameras. The goal in our experiments is to achieve
a minimum field of view of -15 degrees to 75 degrees verically and -135 to 135
degrees horizontally.



