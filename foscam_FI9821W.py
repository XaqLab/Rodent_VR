from numpy import tan

# Parameters for the Foscam FI9821W camera we're using to calibrate the dome

pixel_width = 1280
pixel_height = 720

# Horizontal and vertical fields of view.  These are made up.
theta_x = 1.0
theta_y = theta_x / (float(pixel_width) / float(pixel_height))

# Horizontal and vertical focal lengths in pixels
fpx = (pixel_width/2)/tan(theta_x/2)
fpy = (pixel_height/2)/tan(theta_y/2)

# This is the distance from the intersection of the pitch and yaw rotation axes
# to the center of the lens.  This value is made up.
axes_to_lens = 0.02

# These radial distortion coefficients were calculated by camera_calibration.py
# for the logitech camera.  Need to redo this for the Foscam FI9821W.
distortion_coefficients = [-1.17838063e-18, -9.44769420e-19, 2.13086476e-19]

# completely made up, meters
focal_length = 0.01
