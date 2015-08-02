from numpy import tan

# Parameters for the Logitech C525 web camera we're using to calibrate the dome

pixel_width = 1280
pixel_height = 720

# This horizontal field of view, in radians, was estimated during calibration
theta_x = 1.0
theta_y = theta_x / (float(pixel_width) / float(pixel_height))

distance_to_screen = 3.0
screen_width = 2 * distance_to_screen * tan(theta_x)
screen_height = 2 * distance_to_screen * tan(theta_y)

# measured this with a tape measure, millimeters
height_above_pitch_axis = 0.018
height_above_pitch_axis = 0.0

# These radial distortion coefficients were calculated by camera_calibration.py
distortion_coefficients = [-1.17838063e-18, -9.44769420e-19, 2.13086476e-19]

# completely made up, meters
focal_length = 0.01
focal_length = 0.0
