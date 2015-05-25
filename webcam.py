from numpy import tan

# Parameters for the web camera we're using to calibrate the dome

pixel_width = 1280
pixel_height = 720

# This parameter was estimated during calibration
theta = 0.5

distance_to_screen = 3.0
screen_width = 2 * distance_to_screen * tan(theta)
screen_height = screen_width / (float(pixel_width) / float(pixel_height))
