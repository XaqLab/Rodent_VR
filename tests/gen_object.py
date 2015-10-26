from matplotlib import pyplot as plot
from numpy import array, zeros, uint8, floor, histogram, arange, copy, mean
from numpy.linalg import norm
from random import randint, randrange


def neighbors(pixel):
    """ return a list of the pixels that share a border with this one """
    x, y = pixel
    #neighbors = [[x - 1, y + 1], [x, y + 1], [x + 1, y + 1],
    #             [x - 1, y],                 [x + 1, y],
    #             [x - 1, y - 1], [x, y - 1], [x + 1, y - 1]]
    neighbors = [                [x, y + 1],
                 [x - 1, y],                 [x + 1, y],
                                 [x, y - 1]]
    return neighbors


class PixelObject():
    """ Define a class to enforce the constraint that the object consist of
    contiguous image pixels. """
    def __init__(self, first_pixel):
        """ initialize the list of pixels that the object contains and the
        lists of pixels that are addable """
        self.pixels = [first_pixel]
        self.addables = neighbors(first_pixel)
        self.removables = []


    def copy(self):
        """ return a copy of this pixel object """
        new_object = PixelObject([0, 0])
        new_object.pixels = list(self.pixels)
        new_object.addables = list(self.addables)
        return new_object


    def show(self):
        """ make an image of this pixel object """
        from PIL import Image
        x, y = self.centroid()
        pixels = zeros([2*floor(y), 2*floor(x)], dtype=uint8)
        for pixel in self.pixels:
            x, y = pixel
            pixels[y, x] = 255
        image = Image.fromarray(pixels, mode='L')
        image.show()


    def centroid(self):
        """ return the arithmetic mean of the rows and columns of all the
        pixels in the object """
        x = mean([p[0] for p in self.pixels])
        y = mean([p[1] for p in self.pixels])
        return x, y


    def neighbors_in_object(self, pixel):
        """ determine the neighbors of a pixel that are part of the object """
        neighbors_in_object = []
        for neighbor in neighbors(pixel):
            if neighbor in self.pixels:
                neighbors_in_object.append(neighbor)
        return neighbors_in_object


    def addable(self, pixel):
        """ determine if a pixel can be added to the object """
        if pixel in self.pixels:
            # pixel is already part of the object
            addable = False
        elif len(self.neighbors_in_object(pixel)) == 0:
            # pixel has no neighbor in the object
            addable = False
        else:
            addable = True
        return addable


    def removable(self, pixel):
        """ determine if a pixel can be removed from the object """
        removable = False
        if pixel not in self.pixels:
            removable = False
        elif len(self.neighbors_in_object(pixel)) == 1:
            # pixels with only one neighbor in the object can always be removed
            removable = True
        elif len(self.neighbors_in_object(pixel)) == 2:
            # pixels with two neighbors in the object can be removed if its
            # neighbors are still connected after removal
            neighbor0 = self.neighbors_in_object(pixel)[0]
            neighbors_of_neighbor0 = self.neighbors_in_object(neighbor0)
            neighbors_of_neighbor0.pop(neighbors_of_neighbor0.index(pixel))
            neighbor1 = self.neighbors_in_object(pixel)[1]
            neighbors_of_neighbor1 = self.neighbors_in_object(neighbor1)
            neighbors_of_neighbor1.pop(neighbors_of_neighbor1.index(pixel))
            for neighbor in neighbors_of_neighbor0:
                if neighbor in neighbors_of_neighbor1:
                    removable = True
        return removable


    def add_pixel(self, pixel):
        """ add an addable pixel to the object """
        if pixel in self.addables:
            # add this pixel to the list of pixels in the object
            self.pixels.append(pixel)
            # add this pixel to the list of removable pixels
            self.removables.append(pixel)
            # remove this pixel from the list of addable pixels
            self.addables.pop(self.addables.index(pixel))
            # update the addables and removables lists
            for neighbor in neighbors(pixel):
                if self.addable(neighbor) and pixel not in self.addables:
                    self.addables.append(neighbor)
                if self.removable(neighbor) and pixel not in self.removables:
                    self.removables.append(neighbor)
            pixel_added = True
        else:
            pixel_added = False
        return pixel_added


    def remove_pixel(self, pixel):
        """ remove a pixel from the object """
        if self.removable(pixel):
            # remove this pixel from the object
            self.pixels.pop(self.pixels.index(pixel))
            # add this pixel to the addables list
            self.addables.append(pixel)
            # update the addables and removables lists
            for neighbor in neighbors(pixel):
                if neighbor in self.addables and not self.addable(neighbor):
                    self.addables.pop(self.addables.index(neighbor))
                if neighbor in self.removables and not self.removable(neighbor):
                    self.removables.pop(self.removables.index(neighbor))
            pixel_removed = True
        else:
            pixel_removed = False
        return pixel_removed


def gen_object(centroid, tolerance=1e-6):
    """ Generate an object with a centroid that is within the tolerance
    distance of the given centroid, or die trying.
    """
    x = centroid[0]
    y = centroid[1]
    obj = PixelObject([floor(x), floor(y)])
    difference = norm(centroid - obj.centroid())
    # try adding and removing pixels to minimize the difference between the
    # given centroid and the centroid of our pixel object
    iteratiions = 1
    while iteratiions < 10000 and difference > tolerance:
        iteratiions = iteratiions + 1
        new_obj = obj.copy()
        # add or remove some pixels
        num_pixels = randint(1 - len(new_obj.removables), 10)
        if num_pixels >= 0:
            # add num_pixel + 1 pixels so 0 doesn't go to waste
            for i in range(num_pixels + 1):
                # add a random pixel
                j = randrange(len(new_obj.addables))
                new_obj.add_pixel(new_obj.addables[j])
                new_diff = norm(centroid - new_obj.centroid())
                if new_diff < difference:
                    obj = new_obj.copy()
                    difference = new_diff
        else:
            for i in range(-num_pixels):
                # remove a random pixel
                j = randrange(len(new_obj.removables))
                new_obj.remove_pixel(new_obj.removables[j])
                new_diff = norm(centroid - new_obj.centroid())
                if new_diff < difference:
                    obj = new_obj.copy()
                    difference = new_diff
    print iteratiions
    print difference
    #import pdb; pdb.set_trace()
    return obj


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
