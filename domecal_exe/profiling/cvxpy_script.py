from numpy import pi, sin, cos
from numpy.linalg import norm
from scipy.optimize import fmin_powell
from dome_projection import DomeProjection
from cvxpy import *


def direction_differences(projector_pixels, desired_directions, dome):
    """
    Calculate the sum of the L2 norm of the differences between the desired
    and actual directions.
    """
    #assert len(projector_pixels) == 2*len(desired_directions)

    # find the animal viewing directions for the pixels in projector_pixels
    actual_directions = []
    for n in range(len(desired_directions)):
        row = projector_pixels[2*n]
        col = projector_pixels[2*n + 1]
        actual_direction = dome.dome_display_direction(row, col)[1]
        actual_directions.append(actual_direction)

    value = sum([norm(desired_directions[i] - actual_directions[i])
                 for i in range(len(desired_directions))])
    return value


def find_projector_pixels(directions, dome, pixels=[]):
    """
    Search the projector pixels to find the pixels that minimize the square
    differences between the desired directions and the actual directions.
    """
    if not pixels:
        # no pixel values provided, guess pixels that hit the mirror
        projector_pixel_width = 1280
        projector_pixel_height = 720
        pixels = [projector_pixel_height - 1,
                  projector_pixel_width/2]*len(directions)

    # Find the projector pixels by minimizing the difference between
    # the desired and actual directions.
    arguments = tuple([directions, dome])
    results = fmin_powell(direction_differences, pixels, args=arguments,
                          ftol=1.0, disp=False, full_output=1)
    print results[1]
    #import pdb; pdb.set_trace()
    results = results[0]

    # Sort the final results into pixels
    projector_pixels = []
    for n in range(len(results)/2):
        row = int(round(results[2*n]))
        col = int(round(results[2*n + 1]))
        projector_pixels.append([row, col])

    return projector_pixels


if __name__ == "__main__":
    # make a list of the desired directions for calibration
    calibration_directions = []
    for pitch in [-15, 0, 30, 60]:
    #for pitch in [0, 30, 60]:
        for yaw in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
        #for yaw in [0, 30, 60]:
            x = sin(yaw * pi/180) * cos(pitch * pi/180)
            y = cos(yaw * pi/180) * cos(pitch * pi/180)
            z = sin(pitch * pi/180)
            calibration_directions.append([x, y, z])
    # add straight up
    calibration_directions.append([0, 0, 1])
    print len(calibration_directions)
    
    # run the minimization to find the pixels corresponding to these directions
    dome = DomeProjection()
    x = Variable(2*len(calibration_directions))
    objective = Minimize(direction_differences(x, calibration_directions, dome))
    constraints = [0 <= x, x <= 1280]
    prob = Problem(objective, constraints)
    result = prob.solve()
    print(x.value)

    #pixels = find_projector_pixels(calibration_directions, dome)
    #print
    #print pixels


"""
from cvxpy import *
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)
"""
