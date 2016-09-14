""" Find some initial pixel values for caldev.exe so it doesn't have to do this
every time it's run because find_projector_points runs an optimization that is
SLOW. This could be fixed by coding up an analytical reverse mapping. """

from dome_projection import DomeProjection
from dome_calibration import compute_directions

dome = DomeProjection(projector_pixel_width=1920, projector_pixel_height=1080)

pitch_angles = [60, 30, 0, -15]
yaw_angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
calibration_directions = compute_directions(pitch_angles, yaw_angles)

centroids = dome.find_projector_points(calibration_directions)

pixels = []
for centroid in centroids:
    # convert centroids from (u, v) coordinates to (row, col) coordinates
    col, row = centroid
    row = int(round(row - 0.5))
    col = int(round(col - 0.5))
    pixels.append([row, col])

print pixels
