# Parameters for the web camera we're using to calibrate the dome

pixel_width = 1280
pixel_height = 720


# Field of view measurements
# The screen height and width were calculated based on measurements from webcam
# photos.  These numbers are in inches.

screen_height = 2.375
screen_width = 4.0625
distance_to_screen = 3.0

# This parameter was estimated during calibration
theta = 0.597554159536

[screen_height, screen_width, distance_to_screen] = calc_webcam_FoV(theta)
