from numpy import pi, sin, cos
from numpy.linalg import norm
from scipy.optimize import fmin_powell
from dome_projection import DomeProjection
import cPickle as pickle


def direction_differences(projector_pixels, desired_directions, dome):
    """
    Calculate the sum of the L2 norm of the differences between the desired
    and actual directions.
    """
    assert len(projector_pixels) == 2*len(desired_directions)

    # find the animal viewing directions for the pixels in projector_pixels
    actual_directions = []
    for n in range(len(desired_directions)):
        row = int(projector_pixels[2*n])
        col = int(projector_pixels[2*n + 1])
        actual_direction = dome.dome_display_direction(row, col)[1]
        actual_directions.append(actual_direction)

    value = sum([norm(desired_directions[i] - actual_directions[i])
                             for i in range(len(desired_directions))])
    return value


def find_projector_pixels(directions, dome):
    """
    Search the projector pixels to find the pixels that minimize the square
    differences between the desired directions and the actual directions.
    """
    # get pixel estimates to start with if they exist
    pickle_filename = 'pixels.pkl'
    try:
        with open(pickle_filename, 'rb') as pickle_file:
            x0 = pickle.load(pickle_file)
            #direc0 = pickle.load(pickle_file)
    except IOError:
        # no pixel estimates provided, guess pixels that hit the mirror
        projector_pixel_width = 1280
        projector_pixel_height = 720
        x0 = [projector_pixel_height - 1,
              projector_pixel_width/2]*len(directions)
        #direc0 = None

    # Find the projector pixels by minimizing the difference between
    # the desired and actual directions.
    arguments = tuple([directions, dome])
    results = fmin_powell(direction_differences, x0, args=arguments, xtol=1e-1,
                          ftol=1e0, disp=False, full_output=1)
                          #ftol=1e0, disp=False, full_output=1, direc=direc0)
    #print results[1]
    xopt = results[0]
    direc = results[2]

    # save results to a file for use as the initial guess next time
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(xopt, pickle_file)
        pickle.dump(direc, pickle_file)

    # Sort the final results into pixels
    projector_pixels = []
    for n in range(len(xopt)/2):
        row = int(round(xopt[2*n]))
        col = int(round(xopt[2*n + 1]))
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
    
    # run the minimization to find the pixels corresponding to these directions
    dome = DomeProjection()
    pixels = find_projector_pixels(calibration_directions, dome)

    print
    print pixels
