from PIL import Image
from numpy import array

import sys
import re

if len(sys.argv) == 2:
    filepath = sys.argv[1]
    filename = re.split(r"/",filepath)[-1]
    directory = filepath.replace(filename,"")
    input_image = Image.open(directory + filename)

    pixels = array(input_image)
    for row in range(720):
        for col in range(1280):
            if col == 639 or col == 640:
                pixels[row, col] = 0

    output_image = Image.fromarray(pixels)
    output_image.show()
    output_image.save(directory + "center_line_" + filename,'jpeg')
