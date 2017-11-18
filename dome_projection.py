#!/usr/local/bin/python
# written for python 2.7.8

"""
This class modifies (a.k.a. warps) images from OpenGL for display on a
hemispherical screen in a virtual reality system.  A quarter sphere mirror is
used to allow a single projector to achieve a very wide field of view.  The
goal in our mouse experiments is to achieve a minimum field of view of -15
degrees to 75 degrees verically and -135 to 135 degrees horizontally.  The
OpenGL images are modified to eliminate the distortion introduced by the
quarter sphere mirror and the hemispherical screen.  This is achieved by a
pixel level mapping which sets the RGB values for each projector pixel using
the RGB values of the pixel in the OpenGL image which is in the same direction
from the viewer's point of view.  The viewer in the dome display is the mouse
and the viewers in OpenGL are virtual cameras.  """

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plot
    from matplotlib.backends.backend_pdf import PdfPages

from numpy import array, ones, zeros, dstack, linalg, dot, cross
from numpy import sqrt, sin, cos, tan, pi, arctan
from numpy import arctan2, arccos, where
from numpy import uint8
from random import randint
from PIL import Image
from scipy.optimize import fmin_powell


class NoViewingDirection(Exception):
    """ Exception to handle cases where there is no viewing direction because
    the light from a point in the projector image never reaches the dome. """
    def __init__(self):
         self.value = ("There is no viewing direction for" +
                       "this point in the projector image!")
    def __str__(self):
        return repr(self.value)


class DomeProjection:
    """
    The dome projection class describes the geometry of our dome, spherical
    mirror, and projector along with the geometry of the associated OpenGL
    virtual camera and screen.
    All vectors in the OpenGL world are relative to the virtual camera and
    all vectors in the dome display setup are relative to the center of the
    mirror.
    """
    def __init__(self,
                 screen_height = [1.0, 1.0, 1.0],
                 screen_width = [1.4, 1.4, 1.4],
                 distance_to_screen = [0.5, 0.5, 0.5],
                 pitch = [30, 30, 30],
                 yaw = [-90, 0, 90],
                 image_pixel_height = [200, 200, 200],
                 image_pixel_width = [280, 280, 280],
                 projector_pixel_height = 720,
                 projector_pixel_width = 1280,
                 projector_focal_point = [0, 0.858, -0.052],
                 projector_theta = 0.166,
                 projector_vertical_offset = 0.211,
                 projector_roll = 0.01,
                 mirror_radius = 0.162,
                 dome_center = [0, 0.110, 0.348],
                 dome_radius = 0.601,
                 animal_position = [0, 0.056, 0.575]
                ):

        """
        Parameters:
        ----------------------------
        @param screen_height:
            A list of the heights of each OpenGL screen in arbitrary units.
        @param screen_width:
            A list of the widths of each OpenGL screen in arbitrary units.
        @param distance_to_screen:
            A list of the distances from each OpenGL virtual camera to its
            respective screen in arbitrary units.
        @param pitch:
            A list of the angles between the xy plane and each camera's viewing
            direction in degrees (the pitch in aeronautical terms).
        @param yaw:
            A list of the angles in the xy plane between each camera's viewing
            direction and the y-axis (the yaw in aeronautical terms).
        @param image_pixel_height:
            A list of the number of vertical pixels in each OpenGL image.
        @param image_pixel_width:
            A list of the number of horizontal pixels in each OpenGL image.
        @param projector_pixel_height:
            The number of vertical pixels in the projector image.
        @param projector_pixel_width:
            The number of horizontal pixels in the projector image.
        @param projector_focal_point:
            The (x,y,z) location of the projector's focal point. x should be 0
        @param projector_theta:
            Half of the projector's horizontal field of view, in radians.
        @param projector_vertical_offset:
            A measure of how much the projector's image rises vertically as it
            travels away from the projector.
        @param projector_roll:
            The angle of rotation of the projector. This allows compensation
            for small differences in the orientation of the projector relative
            to the observer.
        @param mirror_radius:
            The radius of the mirror in arbitrary units.
        @param dome_center:
            An (x,y,z) vector from the center of the mirror to the center of
            the dome.
        @param dome_radius:
            The radius of the dome in arbitrary units.
        @param animal_position:
            An (x,y,z) vector from the center of the mirror to the position
            of the animal's eyes.
        """

        #######################################################################
        # Properties passed in as arguments
        #######################################################################
        self._screen_height = screen_height
        self._screen_width = screen_width
        self._distance_to_screen = distance_to_screen
        self._pitch = [i * pi/180.0 for i in pitch]
        self._yaw = [i * pi/180.0 for i in yaw]
        self._image_pixel_height = image_pixel_height
        self._image_pixel_width = image_pixel_width
        self._projector_pixel_height = projector_pixel_height
        self._projector_pixel_width = projector_pixel_width
        self._projector_focal_point = projector_focal_point
        self._projector_theta = projector_theta
        self._projector_vertical_offset = projector_vertical_offset
        self._projector_roll = projector_roll
        self._mirror_radius = mirror_radius
        self._dome_center = dome_center
        self._dome_radius = dome_radius
        self._animal_position = animal_position

        #######################################################################
        # Properties used to share results between method calls
        #######################################################################

        # Start search in the middle of the projector's bottom row of pixels
        self._projector_pixel_row = projector_pixel_height - 1
        self._projector_pixel_col = projector_pixel_width/2

        #######################################################################
        # Properties calculated from arguments
        #######################################################################
        """
        Calculate the vectors to the center of each screen along with unit
        vectors that indicate the directions for increasing row and column
        numbers. These vectors will make it very easy to calculate the row and
        column of a pixel given the direction from the viewer to that pixel.
        """
        self._vector_to_screen = []
        self._row_vector = []
        self._col_vector = []
        for screen in range(len(distance_to_screen)):
            yaw = self._yaw[screen]
            pitch = self._pitch[screen]
            x = (distance_to_screen[screen] * sin(yaw) * cos(pitch))
            y = (distance_to_screen[screen] * cos(yaw) * cos(pitch))
            z = distance_to_screen[screen] * sin(pitch)
            self._vector_to_screen.append(array([x, y, z]))
            # Create a column vector that will be used as a basis vector to
            # find the column number of pixels in images from this camera.
            col_vector = cross(self._vector_to_screen[screen],
                               array([0, 0, 1]))
            col_vector = col_vector / linalg.norm(col_vector)
            self._col_vector.append(col_vector)
            # Create a row vector that will be used as a basis vector to
            # find the row number of pixels in images from this camera.
            row_vector = cross(self._vector_to_screen[screen], col_vector)
            row_vector = row_vector / linalg.norm(row_vector)
            self._row_vector.append(row_vector)

        """
        Make the necessary preparations to calculate the unit vectors
        (directions) from the animal inside the dome towards the projection
        of each projector pixel on to the dome.  But delay this time consuming
        calculation until it is necessary.
        """
        # Calculate projector image height and width.
        # This calculation assumes the projector image has the same width at
        # the top and bottom, i.e. the projector's keystone adjustment is set
        # properly.
        self._projector_distance_to_image = 1.0
        self._projector_image_width = 2*arctan(self._projector_theta)
        projector_pixel_width = float(self._projector_pixel_width)
        projector_pixel_height = float(self._projector_pixel_height)
        aspect_ratio = projector_pixel_width/projector_pixel_height
        self._projector_image_height = self._projector_image_width/aspect_ratio

        # Setup properties that will be calculated if warp_image_for_dome is
        # called.
        self._projector_mask = []
        self._animal_view_directions = []
        self._contributing_pixels = []

        # Setup properties that will be calculated if _unwarp_image is called.
        self._camera_view_directions = []



    ###########################################################################
    # Class methods
    ###########################################################################

    def warp_image_for_dome(self, images):
        """
        Take an RGB input image intended for display on a flat screen and
        produce an image for the projector that removes the distortions caused
        by projecting the image onto the dome using a spherical mirror.
        """
        if self._contributing_pixels == []:
            """
            Build lists of the OpenGL image pixels that contribute to each
            projector pixel.
            """
            self._contributing_pixels = self._calc_contributing_pixels()

        pixels = []
        for i, image in enumerate(images):
            assert image.size == (self._image_pixel_width[i],
                                  self._image_pixel_height[i])
            pixels.append(array(image))

        warped_pixels = zeros([self._projector_pixel_height,
                               self._projector_pixel_width, 3], dtype=uint8)
        for row in range(self._projector_pixel_height):
            for col in range(self._projector_pixel_width):
                pixel_value = zeros(3) # for a black background
                #pixel_value = 255*ones(3) # for a white background
                for pixel in self._contributing_pixels[row][col]:
                    pixel_value += pixels[pixel[0]][pixel[1], pixel[2]]
                n = len(self._contributing_pixels[row][col])
                if n > 0:
                    pixel_value = pixel_value/n
                warped_pixels[row][col] = array(pixel_value, dtype=uint8)

        return Image.fromarray(warped_pixels, mode='RGB')


    def _unwarp_image(self, warped_image):
        """
        Take an image intended for projection onto the dome and reconstruct the
        images used to make it.
        """
        assert warped_image.size == (self._projector_pixel_width,
                                     self._projector_pixel_height)

        if self._contributing_pixels == []:
            """
            Build lists of the OpenGL image pixels that contribute to each
            projector pixel.
            """
            self._contributing_pixels = self._calc_contributing_pixels()

        if self._camera_view_directions == []:
            """
            Calculate the unit vectors (aka directions) that point from
            each OpenGL virtual camera towards all of the pixels in its image.
            """
            self._camera_view_directions = \
               [flat_display_directions(self._screen_height[i],
                                         self._screen_width[i],
                                         self._image_pixel_height[i],
                                         self._image_pixel_width[i],
                                         self._distance_to_screen[i],
                                         pitch = self._pitch[i],
                                         yaw = self._yaw[i]) 
                 for i in range(len(self._screen_height))]

        warped_pixels = array(warped_image)
        pixels = [zeros([self._image_pixel_height[i],
                         self._image_pixel_width[i], 3], dtype=uint8)
                  for i in range(len(self._image_pixel_height))]
        for image in range(len(self._image_pixel_height)):
            row = 0
            while row < self._image_pixel_height[image]:
                columns = range(self._image_pixel_width[image])
                for col in columns:
                    """
                    For each image pixel, find the closest projector pixel
                    and use its RGB values.
                    Go through the pixels in a serpentine pattern so that the
                    current pixel is always close to the last pixel.  This way the
                    search algorithm can use the last result as its starting point.
                    """
                    projector_pixel = self._find_closest_projector_pixel(image,
                                                                         row, col)
                    pp_row = projector_pixel[0]
                    pp_col = projector_pixel[1]
                    if self._projector_mask[pp_row, pp_col] == 1:
                        pixels[image][row, col] = warped_pixels[pp_row, pp_col]
                row = row + 1
                columns.reverse()

        return [Image.fromarray(pixels[i], mode='RGB') for i in
                range(len(pixels))]


    def _calc_dome_display_directions(self):
        """
        Return the unit vectors (directions) from the viewer inside the dome
        towards the projection of each projector pixel on to the dome.
        """
        self._projector_mask = zeros([self._projector_pixel_height,
                                      self._projector_pixel_width],
                                     dtype=uint8)
        self._animal_view_directions = zeros([self._projector_pixel_height,
                                              self._projector_pixel_width, 3])
        for row in range(self._projector_pixel_height):
            for column in range(self._projector_pixel_width):
                u = column + 0.5
                v = row + 0.5
                try:
                    direction = self.dome_display_direction(u, v)
                    self._projector_mask[row, column] = 1
                    self._animal_view_directions[row, column] = direction
                except NoViewingDirection:
                    # This projector pixel did not hit the mirror or it hit
                    # the mirror and missed the dome
                    self._projector_mask[row, column] = 0
                    self._animal_view_directions[row, column] = None


    def dome_display_direction(self, u, v):
        """
        Return the unit vector (direction) from the viewer inside the dome
        towards the projection of the specified (u, v) coordinate from the
        projector image on to the dome.  All vectors used in these calculations
        are relative to the center of the hemispherical mirror.  The projector
        is on the positive y-axis (but projecting in the -y direction) and its
        projected image is assumed to be horizontally centered on the mirror.
        """

        # Calculate the direction of light emanating from the projector pixel.
        # Yaw is Pi, or 180 degrees, because projection is in -y direction.
        projector_pixel_direction = \
            flat_display_direction(u, v, self._projector_image_height,
                                   self._projector_image_width,
                                   self._projector_pixel_height,
                                   self._projector_pixel_width,
                                   self._projector_distance_to_image,
                                   self._projector_vertical_offset,
                                   pitch=0,
                                   yaw=pi,
                                   roll=self._projector_roll)

        """
        Complete the triangle consisting of:
            1.  the vector from the center of the mirror to the projector's
                focal point (completely specified)
            2.  the vector from the projector's focal point to the mirror for
                the given projector pixel (known direction, unknown length)
            3.  the vector from the center of the mirror to the point on the
                mirror where the vector in 2 hits the mirror (known length,
                unknown direction)
        Vector 3 is normal to the mirror's surface at the point of reflection
        and is used to calculate the direction of the reflected light.
        """
        # solve quadratic equation for y-component of vector 2
        px = self._projector_focal_point[0]
        py = self._projector_focal_point[1]
        pz = self._projector_focal_point[2]
        pdx = projector_pixel_direction[0]
        pdy = projector_pixel_direction[1]
        pdz = projector_pixel_direction[2]
        a = pdx**2 + pdy**2 + pdz**2
        b = 2*px*pdx + 2*py*pdy + 2*pz*pdz
        c = px**2 + py**2 + pz**2 - self._mirror_radius**2
        incident_light_vector = 0
        """
        The vector will intersect the sphere twice.  Pick the root
        for the shorter vector.
        """
        d_squared = b**2 - 4*a*c
        if d_squared < 0:
            # raise an exception for points that miss the mirror
            raise NoViewingDirection()
        """
        For projector pixels that hit the mirror, calculate the
        incident light vector.
        """
        d = sqrt(d_squared)
        r = min([(-b + d) / (2*a), (-b - d) / (2*a)])
        x = r*pdx
        y = r*pdy
        z = r*pdz
        incident_light_vector = array([x, y, z])

        mirror_radius_vector = (self._projector_focal_point +
                                 incident_light_vector)
        mirrorUnitNormal = mirror_radius_vector / self._mirror_radius

        """
        Use the incident_light_vector and the mirrorUnitNormal to calculate
        the direction of the reflected light.
        """
        reflectedLightDirection = zeros([3])
        u = mirrorUnitNormal
        reflectedLightVector = \
            -2*dot(incident_light_vector, u)*u \
            + incident_light_vector
        reflectedLightDirection = \
            reflectedLightVector/linalg.norm(reflectedLightVector)

        """
        Complete the triangle again to get the reflected light vector.
        The known vector is from the center of the dome to the reflection
        point on the mirror (calculated as mirror_radius_vector - dome_center)
        and the length of the vector with unknown direction is the dome radius.
        """
        # solve quadratic for the length of the reflected light vector
        rpx = mirror_radius_vector[0] - self._dome_center[0]
        rpy = mirror_radius_vector[1] - self._dome_center[1]
        rpz = mirror_radius_vector[2] - self._dome_center[2]
        rldx = reflectedLightDirection[0]
        rldy = reflectedLightDirection[1]
        rldz = reflectedLightDirection[2]
        a = rldx**2 + rldy**2 + rldz**2
        b = 2*rpx*rldx + 2*rpy*rldy + 2*rpz*rldz
        c = rpx**2 + rpy**2 + rpz**2 - self._dome_radius**2
        reflected_light_vector = zeros([3])

        # Take the solution with positive vector length which is the reflected
        # light.  The vector with the negative length is directed in the
        # opposite direction.
        d_squared = b**2 - 4*a*c
        if d_squared < 0:
            # raise an exception for reflections that miss the dome
            raise NoViewingDirection()
        d = sqrt(d_squared)
        r = max([(-b + d) / (2*a),
                 (-b - d) / (2*a)])
        x = r*rldx
        y = r*rldy
        z = r*rldz
        reflected_light_vector = [x, y, z]

        """
        Now use the vector of the reflected light, reflection position on the
        mirror, and animal position to calculate the animal's viewing direction
        for the projector pixel.
        """
        # For projector pixels that hit the mirror,
        # calculate the view direction for the animal.
        animal_view_vector = (reflected_light_vector
                              + mirror_radius_vector
                              - self._animal_position)
        magnitude = linalg.norm(animal_view_vector)
        animal_view_direction = animal_view_vector/magnitude

        return animal_view_direction


    def _calc_contributing_pixels(self):
        """
        Use the direction for each projector pixel to calculate the nearest
        OpenGL pixel and use its RGB values for the projector pixel.  
        """

        # Calculate self._projector_mask and self._animal_view_directions
        self._calc_dome_display_directions()

        # This 2D list of lists contains the list of OpenGL pixels
        # that contribute to each projector pixel.
        contributing_pixels = \
            [[[] for i in range(self._projector_pixel_width)]
             for j in range(self._projector_pixel_height)]
        # Calculate center pixels, pixel spacing and direction vectors for each
        # screen.
        screens = range(len(self._distance_to_screen))
        row_center = [(self._image_pixel_height[i] - 1) / 2.0 for i in screens]
        row_spacing = [float(self._screen_height[i])
                            / self._image_pixel_height[i] for i in screens]
        col_center = [(self._image_pixel_width[i] - 1) / 2.0 for i in screens]
        col_spacing = [float(self._screen_width[i])
                            / self._image_pixel_width[i] for i in screens]
        vector_to_screen = self._vector_to_screen
        distance_to_screen = [linalg.norm(v) for v in vector_to_screen]
        direction_to_screen = [vector_to_screen[i] / distance_to_screen[i]
                               for i in screens]
        for row in range(self._projector_pixel_height):
            for col in range(self._projector_pixel_width):
                if self._projector_mask[row, col] == 1:
                    """
                    For each projector pixel that hits the mirror, use the
                    direction in which the animal sees this pixel to find the
                    OpenGL pixel which has the closest direction from the
                    virtual camera's point of view.  The RGB values of this
                    OpenGL pixel will be used for this projector pixel.
                    """
                    # Calculate the magnitude required for a vector in this
                    # direction to hit the nearest OpenGL screen.
                    direction = self._animal_view_directions[row, col]
                    # Find the nearest OpenGL screen
                    screen = 0
                    direction_dot_direction_to_screen = 0
                    for i in screens:
                        dp = direction.dot(direction_to_screen[i])
                        if dp > direction_dot_direction_to_screen:
                            direction_dot_direction_to_screen = dp
                            screen = i
                    if direction_dot_direction_to_screen > 0:
                        """
                        Ignore cases where the dot product of direction and
                        direction_to_screen is negative.  These cases project
                        the image to a second, incorrect location on the dome.
                        """
                        # Calculate the full vector from the unit vector in this
                        # direction and use it to find the row and column numbers
                        # of the OpenGL pixel.
                        magnitude = (distance_to_screen[screen]
                                     / direction_dot_direction_to_screen)
                        x = magnitude * direction[0]
                        y = magnitude * direction[1]
                        z = magnitude * direction[2]
                        on_screen_vector = (array([x, y, z])
                                            - vector_to_screen[screen])
                        r = int(round(row_center[screen]
                             + on_screen_vector.dot(self._row_vector[screen])
                             / row_spacing[screen]))
                        c = int(round(col_center[screen]
                             + on_screen_vector.dot(self._col_vector[screen])
                             / col_spacing[screen]))
                        # make sure the pixel is inside the OpenGL image
                        if (r >= 0 and r < self._image_pixel_height[screen]
                            and c >= 0 and c < self._image_pixel_width[screen]):
                            contributing_pixels[row][col].append([screen, r, c])

        return contributing_pixels


    def _find_closest_projector_pixel(self, image, row, col):
        """
        For the OpenGL image pixel specified by row and col use the directions
        in self._camera_view_directions and self._animal_view_directions to find the
        projector pixel which has the closest direction and return its row and
        column in a list.  This is done using a search method rather than
        calculating the dot product for every projector pixel.
        """
        camera_direction = self._camera_view_directions[image][row, col]

        # Start with the last projector pixel
        r = self._projector_pixel_row
        c = self._projector_pixel_col

        while self._projector_mask[r, c] == 0:
            # if the pixel doesn't hit the mirror, find one that does
            r = randint(self._projector_pixel_height)
            c = randint(self._projector_pixel_width)

        animalDirection = self._animal_view_directions[r, c]

        # Calculate dot product of this OpenGL pixel
        # with the last projector pixel.
        dp = dot(camera_direction, animalDirection)

        # Calculate dot product of this OpenGL pixel with the
        # neighboring projector pixels.
        [neighbor_dps, neighbors] = \
            self._calc_neighbor_dot_products(r, c, camera_direction)

        while max(neighbor_dps) > dp:
            """
            If the dot product with one of the neighboring projector pixels is
            larger then update r and c to that pixel and check its neighbors.
            Repeat until all neighbors have smaller (or equal) dot products.
            """
            dp = max(neighbor_dps)
            [r, c] = neighbors[neighbor_dps.index(dp)]
            [neighbor_dps, neighbors] = \
                self._calc_neighbor_dot_products(r, c, camera_direction)

        # Save projector pixel for next call
        self._projector_pixel_row = r
        self._projector_pixel_col = c
        return [r, c]


    def _calc_neighbor_dot_products(self, row, col, camera_direction):
        """
        For the projector pixel specified by row and col, calculate the dot
        product of all its neighbors with the given camera direction.  Return
        a list containing a list of the dot products and a list of the row and
        column for each corresponding pixel.
        """
        neighbors = []
        neighbor_dps = []

        # find neighbors
        row_above = [[-1, -1], [-1, 0], [-1, 1]]
        beside = [[0, -1], [0, 1]]
        row_below = [[1, -1], [1, 0], [1, 1]]
        offsets = row_above + beside + row_below
        for [dr, dc] in offsets:
            if row + dr >= 0 and row + dr < self._projector_pixel_height:
                if col + dc >= 0 and col + dc < self._projector_pixel_width:
                    neighbors.append([row + dr, col + dc])

        # calculate neighbor dot products
        for neighbor in neighbors:
            neighbor_direction = \
                self._animal_view_directions[neighbor[0], neighbor[1]]
            neighbor_dps.append(dot(camera_direction, neighbor_direction))

        return [neighbor_dps, neighbors]


    def _save_projector_mask_image(self, filename):
        """
        Save an image of the projector mask. This projector mask is 1 for
        projector pixels that hit the mirror and 0 for pixels that miss the
        mirror.
        """

        image = Image.fromarray(array(255*self._projector_mask,
                                     dtype=uint8), mode='L')
        image.save(filename, "png")


    def get_parameters(self):
        """ return a dictionary of parameters """
        parameters = dict(image_pixel_width = self._image_pixel_width,
                          image_pixel_height = self._image_pixel_height,
                          projector_pixel_width = self._projector_pixel_width,
                          projector_pixel_height = self._projector_pixel_height,
                          projector_focal_point = self._projector_focal_point,
                          projector_theta = self._projector_theta,
                          projector_vertical_offset = self._projector_vertical_offset,
                          projector_roll = self._projector_roll,
                          mirror_radius = self._mirror_radius,
                          dome_center = self._dome_center,
                          dome_radius = self._dome_radius,
                          animal_position = self._animal_position)
        return parameters


    def _direction_differences(self, projector_points, desired_directions):
        """
        Calculate the sum of the square differences between the desired and actual
        directions.
        """
        assert len(projector_points) == 2*len(desired_directions)
    
        # find the viewing directions for the points in projector_points
        actual_directions = []
        for n in range(len(desired_directions)):
            u = projector_points[2*n]
            v = projector_points[2*n + 1]
            try:
                actual_direction = self.dome_display_direction(u, v)
            except NoViewingDirection:
                # For each point that has no viewing direction, set its actual
                # viewing direction to the opposite of the desired direction.
                # This will produce the worst possible result for these points
                # thereby encouraging the minimization routine to look
                # elsewhere.
                actual_direction = -array(desired_directions[n])
            actual_directions.append(actual_direction)
    
        diffs = array(desired_directions) - array(actual_directions)
        value = sum([linalg.norm(diff) for diff in diffs])
        return value


    def find_projector_points(self, directions, points=[]):
        """
        Search the projector image to find the (u,v) coordinates that minimize
        the square differences between the desired directions and the actual
        directions.
        """
        arguments = tuple([directions])
        if points:
            # flatten points to make x0 for fmin_powell
            x0 = [coordinate for point in points for coordinate in point]
            # We have estimates of the points' coordinates so use loose
            # tolerances to get the result faster.
            results = fmin_powell(self._direction_differences, x0,
                                  args=arguments, disp=True, full_output=1)
                                  #xtol=0.1, ftol=1.0)
        else:
            # no point coordinates provided so guess points that hit the mirror
            # and use tighter tolerances to get an accurate result
            x0 = [self._projector_pixel_height - 1,
                  self._projector_pixel_width/2]*len(directions)
            results = fmin_powell(self._direction_differences, x0,
                                  args=arguments, disp=False, full_output=1)
        xopt = results[0]
        fopt = results[1]
    
        # Sort the final results into points
        projector_points = []
        for n in range(len(xopt)/2):
            u = xopt[2*n]
            v = xopt[2*n + 1]
            projector_points.append([u, v])
    
        return projector_points


###############################################################################
# Functions called by class methods
###############################################################################

def flat_display_directions(screen_height, screen_width, pixel_height,
                            pixel_width, distance_to_screen,
                            vertical_offset = 0, pitch = 0, yaw = 0):
    """
    Return unit vectors that point from the viewer towards each pixel
    """
    directions = zeros([pixel_height, pixel_width, 3])
    for row in range(pixel_height):
        for column in range(pixel_width):
            u = column + 0.5
            v = row + 0.5
            directions[row, column] = \
                    flat_display_direction(u, v, screen_height,
                                           screen_width, pixel_height,
                                           pixel_width, distance_to_screen,
                                           vertical_offset, pitch, yaw)

    return directions


def flat_display_direction(u, v, screen_height, screen_width,
                           pixel_height, pixel_width, distance_to_screen,
                           vertical_offset = 0, pitch = 0, yaw = 0, roll = 0):
    """
    Return a unit vector that points from the viewer towards the specified
    (u,v) coordinate on a flat screen display.  The display is along the
    positive y-axis relative to the viewer.  The positive x-axis is to the
    viewer's right and the positive z-axis is up.  The vertical_offset
    parameter shifts the screen in the positive z-direction.  Pitch is the
    angle between the vector from the origin to the center of the image and
    the x-y plane.  Yaw is the angle of ratation in the x-y plane.  It is
    positive to the right and negative to the left.
    """
    # calculate the vector to the center of the screen
    x = distance_to_screen * sin(yaw) * cos(pitch)
    y = distance_to_screen * cos(yaw) * cos(pitch)
    z = distance_to_screen * sin(pitch)
    vector_to_screen = array([x, y, z])
    # unit vector that points in the direction of increasing column number
    col_vector = cross(vector_to_screen,
                       array([sin(roll)*cos(yaw),
                              sin(roll)*sin(yaw),
                              cos(roll)]))
    col_vector = col_vector / linalg.norm(col_vector)
    # unit vector that points in the direction of increasing row number
    row_vector = cross(vector_to_screen, col_vector)
    row_vector = row_vector / linalg.norm(row_vector)
    # scale the column unit vector to reach the given column
    col_vector = ((float(screen_width) / pixel_width) *
                  (u - pixel_width/2.0) * col_vector)
    # scale the row unit vector to reach the given row
    row_vector = (float(screen_height) / pixel_height *
                  (v - pixel_height/2.0) * row_vector) 
    # build the vertical offset vector
    offset_vector = array([0, 0, vertical_offset])
    # build the vector to the pixel specified by row and column
    vector_to_pixel = (vector_to_screen + row_vector + col_vector +
                       offset_vector)
    # normalize to get the direction
    direction = vector_to_pixel / linalg.norm(vector_to_pixel)

    return direction

