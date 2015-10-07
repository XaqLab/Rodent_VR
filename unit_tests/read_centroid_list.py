from numpy import array
from numpy.linalg import norm

import sys
sys.path.append("..") # in case this file is run from .
sys.path.append(".")  # in case this file is run from ..
from dome_calibration import read_centroid_list

test_data = [
    [366.5, 639.5],
    [354.5, 564.5],
    [323.5, 532.5],
    [445.5, 639.5],
    [424.5, 481.5],
    [361.5, 405.5],
    [573.5, 639.5],
    [545.5, 391.5],
    [459.5, 247.5]]

required_accuracy = 1e-6
centroid_list = read_centroid_list("centroid_list.txt")
for i in range(len(test_data)):
    """ Make sure we get the right numbers from the file. """
    centroid1 = array(test_data[i])
    centroid2 = array(centroid_list[i])
    if norm(centroid2 - centroid1) > required_accuracy:
        print "Unexpected centroid read from file!"
        print "Expected centroid:", centroid1
        print "Centroid read from file:", centroid2
print "Done with read_centroid_list test"

