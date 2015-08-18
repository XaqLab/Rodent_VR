# Import statements
from numpy import array, float32

# Camera properties found using OpenCV
pixel_width = 1280
pixel_height = 720
fpx = 1199.762731
fpy = 1263.368285
ppx = 598.446870
ppy = 345.534594
distortion_coefficients = array([[ -0.121147, -0.015633, -0.002517, -0.010681, 0.085236 ]])

matrix = array([[ 1199.763,    0.000,  598.447 ],
                [    0.000, 1263.368,  345.535 ],
                [    0.000,    0.000,    1.000 ]], dtype=float32)

# Camera properties found NOT using OpenCV

# This is the distance, in meters, from the intersection of the
# pitch and yaw rotation axes to the center of the lens.  This value
# was measured crudely with a tape measure.
axes_to_lens = 0.035

# This is the focal distance, in meters, of the camera's lens.  I just
# guessed a value.
focal_length = 0.001
