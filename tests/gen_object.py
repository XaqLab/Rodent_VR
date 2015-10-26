from matplotlib import pyplot as plot
from numpy import array, zeros, uint8, floor, histogram, arange, copy
from numpy.linalg import norm
from random import randint


class PixelObject():
    """ Define a class to enforce the constraint that the object consist of
    contiguous image pixels. """
    def __init__(first_pixel):
        """ initialize the list of pixels that the object contains and the list
        of pixels that are eligible to be added to the object """
        self.pixels = [first_pixel]
        x, y = first_pixel
        self.eligible = neighbors(first_pixel)


    def centroid(self):
        """ return the arithmetic mean of the rows and columns of all the
        pixels in the object """
        x = mean([p[0] for p in self.pixels])
        y = mean([p[1] for p in self.pixels])
        return x, y


    def add_pixel(self, pixel):
        """ add an eligible pixel to the object """
        if pixel in self.eligible:
            # add this pixel to the object
            self.pixels.append(pixel)
            # remove this pixel from the eligible list
            self.eligible.pop(pixel)
            # add the neighbors of this pixel to the eligible list if they are
            # not yet on it
            for neighbor in neighbors(pixel):
                if neighbor not in self.eligible:
                    self.eligible.append(neighbor)


def gen_object(centroid, width, tolerance=1e-6):
    """ Find a distribution of pixels over columns and rows that has the
    given centroid.  Assume a solution of the form (for width=2):
        x = (a1*floor(x) + a2*(floor(x) + 1))/(a1 + a2)
        y = (b1*floor(y) + b2*(floor(y) + 1))/(b1 + b2)
        a1 + a2 = b1 + b2

    So I'm approximating floating point numbers by a weighted average of
    integers that have values close to them.
    """
    x = centroid[0]
    y = centroid[1]
    x_integers = array(range(width)) - (width - 1)/2 + floor(x)
    y_integers = array(range(width)) - (width - 1)/2 + floor(y)
    x_weights = array(x_integers == floor(x), dtype='int')
    y_weights = array(y_integers == floor(y), dtype='int')
    x_value = sum(x_weights*x_integers)/float(sum(x_weights))
    y_value = sum(y_weights*y_integers)/float(sum(y_weights))
    difference = norm(centroid - array(x_value, y_value))
    # try different weights to minimize the difference between the true
    # centroid and our approximation
    new_x_weights = copy(x_weights)
    new_y_weights = copy(y_weights)
    iteratiions = 1
    while iteratiions < 1000 and difference > tolerance:
        iteratiions = iteratiions + 1
        for i in range(len(x_weights)):
            # pick a random set of weights that is consistent with an actual
            # image, i.e.  sum(new_x_weights) = sum(new_y_weights)
            new_x_weights[i] = randint(1, width)
            new_y_weights[i] = randint(1, width)



            new_x_value = sum(new_x_weights*x_integers)/float(sum(new_x_weights))
            new_diff = abs(x - new_value)
            if new_diff < difference:
                x_weights = copy(new_x_weights)
                y_weights = copy(new_y_weights)
                difference = new_diff
    #import pdb; pdb.set_trace()
    print iteratiions
    return x_weights, x_integers, difference


def random_numbers():
    n = 10000
    m = 10
    x = zeros(n)
    for i in range(n):
        x[i] = randint(0, m)
    xs = array(range(0, m + 1))
    bins = array(range(0, m + 2)) - 0.5
    print "bins", bins
    p = histogram(x, bins, density=True)[0]
    print "p", p
    plot.figure()
    axes = plot.subplot(111)
    #import pdb; pdb.set_trace()
    axes.plot(xs, p)
    axes.set_ylim([0, 2/float(m)])
    plot.show()


if __name__ == "__main__":
    random_numbers()
