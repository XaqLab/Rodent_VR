""" Stopping this for now because cvxopt requires derivatives which would be
non-trivial to compute """


from numpy import pi, sin, cos
from scipy.optimize import fmin_powell
from dome_projection import DomeProjection
from cvxopt.solvers import cp


class FindPixels(dome):
    """ This class sets up the minimization problem used to find projector
    pixels that correspond to a viewing direction inside the dome.  It sets up
    this problem in a way that allows the use of the non-linear solver from
    CVXOPT.  This solver does not have a mechanism for passing in arguments
    besides the one being optimized so I am making this class so those
    arguments can be stored as class properties. """
    def __init__(self, dome):
        """ Save the dome geometry, setup the calibration directions and setup
        pixels from which the minimization can start. """
        # save the dome geometry
        self.dome = dome
        # make a list of the desired directions for calibration
        self.calibration_directions = []
        for pitch in [-15, 0, 30, 60]:
            for yaw in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
                x = sin(yaw * pi/180) * cos(pitch * pi/180)
                y = cos(yaw * pi/180) * cos(pitch * pi/180)
                z = sin(pitch * pi/180)
                self.calibration_directions.append([x, y, z])
        # add straight up
        self.calibration_directions.append([0, 0, 1])
    
        # guess pixels that hit the mirror
        self.x0 = [projector_pixel_height - 1,
                   projector_pixel_width/2]*len(self.calibration_directions)


    def direction_differences(self, projector_pixels):
        if projector_pixels == None:
            """ Tell the solver what the dimensions of the problem are and
            where to start """
            m = 0 # the number of non-linear constraints
            return_value = (m, self.x0)
        else:
            """
            Calculate the sum of the L2 norm of the differences between the desired
            and actual directions.
            """
            desired_directions = self.calibration_directions
            assert len(projector_pixels) == 2*len(desired_directions)
        
            # find the animal viewing directions for the pixels in projector_pixels
            return_value = 0
            for n in range(len(desired_directions)):
                row = int(projector_pixels[2*n])
                col = int(projector_pixels[2*n + 1])
                actual_direction = dome.dome_display_direction(row, col)[1]
                actual_directions.append(actual_direction)
        
            value = sum([linalg.norm(desired_directions[i] - actual_directions[i])
                                     for i in range(len(desired_directions))])
            return value

        return return_value


def find_projector_pixels(directions, dome, pixels=[]):
    """
    Search the projector pixels to find the pixels that minimize the square
    differences between the desired directions and the actual directions.
    """
    if not pixels:
        # no pixel values provided, guess pixels that hit the mirror
        pixels = [projector_pixel_height - 1,
                  projector_pixel_width/2]*len(directions)

    # Find the projector pixels by minimizing the difference between
    # the desired and actual directions.
    arguments = tuple([directions, dome])
    results = cp(direction_differences, pixels, args=arguments,
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
    # run the minimization to find the pixels corresponding to these directions
    dome = DomeProjection()
    pixels = dome.find_projector_pixels(calibration_directions)
    print
    print pixels


