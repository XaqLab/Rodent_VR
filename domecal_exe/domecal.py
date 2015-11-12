# import win32 stuff
import winxpgui as win32gui
import win32api
import win32con

# import dome projection stuff
from dome_projection import DomeProjection
from dome_projection import calc_projector_images

NUM_PARAMETERS = 10
PROJECTOR_PIXEL_WIDTH = 1280
PROJECTOR_PIXEL_HEIGHT = 720

# Color constants
BLACK = win32api.RGB(0, 0, 0)
RED = win32api.RGB(128, 0, 0)
GREEN = win32api.RGB(0, 128, 0)
BLUE = win32api.RGB(0, 0, 128)
YELLOW = win32api.RGB(128, 128, 0)
CYAN = win32api.RGB(0, 128, 128)
PURPLE = win32api.RGB(128, 0, 128)
WHITE = win32api.RGB(128, 128, 128)
pixel_colors = [WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE,
                WHITE, WHITE, WHITE, WHITE, WHITE]

# key codes not found in win32con
VK_0 = 0x30
VK_1 = 0x31
VK_2 = 0x32
VK_3 = 0x33
VK_4 = 0x34
VK_5 = 0x35
VK_6 = 0x36
VK_7 = 0x37
VK_8 = 0x38
VK_9 = 0x39
VK_F = 0x46
VK_O = 0x4f
VK_S = 0x53
VK_OEM_PLUS = 0xbb
VK_OEM_MINUS = 0xbd

# window styles
regular_window = (win32con.WS_THICKFRAME | win32con.WS_VISIBLE |
                  win32con.WS_SYSMENU | win32con.DS_SETFONT |
                  win32con.WS_MINIMIZEBOX)
fullscreen_window = (win32con.WS_VISIBLE |
                     win32con.DS_SETFONT)
window_styles = {"regular":regular_window,
                 "fullscreen":fullscreen_window}

class MainWindow():
    """ The main application window class """

    def __init__(self):
        dome = DomeProjection()
        self._current_parameter = 0
        # initialize parameter names
        self._parameter_names = ["Mirror Radius",
                                 "Dome Radius",
                                 "Dome y-coordinate",
                                 "Dome z-coordinate",
                                 "Animal y-coordinate",
                                 "Animal z-coordinate",
                                 "Projector horizontal field of view",
                                 "Projector vertical throw",
                                 "Projector y-coordinate",
                                 "Projector z-coordinate"]
        # initialize the increment for each parameter
        self._increments = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2,
                            1e-2, 1e-2]
        # initialize parameter values
        self._parameter_values = []
        self._parameter_values.append(dome.get_mirror_radius())
        self._parameter_values.append(dome.get_dome_radius())
        dome_y, dome_z = dome.get_dome_position()
        self._parameter_values.extend([dome_y, dome_z])
        animal_y, animal_z = dome.get_animal_position()
        self._parameter_values.extend([animal_y, animal_z])
        self._parameter_values.extend(dome.get_frustum_parameters())
        # guess some initial projector pixel values
        self._pixels = []
        for pitch in [-15, 0, 30, 60]:
            for yaw in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
                row = -1*(pitch + 15) + PROJECTOR_PIXEL_HEIGHT - 1
                col = -4*(yaw) + PROJECTOR_PIXEL_WIDTH/2
                self._pixels.append([row, col])
        # add straight up
        row = -1*(90 + 15) + PROJECTOR_PIXEL_HEIGHT - 1
        col = -1*(0) + PROJECTOR_PIXEL_WIDTH/2
        self._pixels.append([row, col])
        # pixels saved after successful convergence
        self._pixels = [ \
                [379, 1244], [643, 1150], [642, 1000],
                [664, 831], [667, 639], [663, 445],
                [643, 275], [618, 143], [404, 22],
                [472, 1116], [535, 1077], [570, 958],
                [589, 810], [597, 640], [591, 469],
                [570, 318], [534, 210], [464, 173],
                [368, 938], [425, 931], [461, 857],
                [481, 759], [486, 640], [480, 521],
                [450, 428], [422, 354], [369, 342],
                [332, 768], [356, 782], [375, 748],
                [401, 704], [404, 640], [399, 577],
                [375, 529], [350, 508], [327, 516],
                [341, 639]]
        # find the projector pixels for the calibration directions using the
        # default parameter values
        self._pixels = dome.find_projector_pixels(dome.calibration_directions,
                                                  self._pixels)
        # setup the window
        self._name = "MainWindow"
        self._title = "Dome Calibration"
        self._width = PROJECTOR_PIXEL_WIDTH
        self._height = PROJECTOR_PIXEL_HEIGHT
        self._style = window_styles["regular"]
        self._fullscreen = False
        win32gui.InitCommonControls()
        self._hinstance = win32gui.dllhandle
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = self._name
        wc.style =  win32con.CS_GLOBALCLASS|win32con.CS_VREDRAW | win32con.CS_HREDRAW
        wc.hCursor = win32gui.LoadCursor( 0, win32con.IDC_ARROW )
        wc.lpfnWndProc = self.window_process
        try:
            classAtom = win32gui.RegisterClass(wc)
        except win32gui.error, err_info:
            if err_info.winerror!=winerror.ERROR_CLASS_ALREADY_EXISTS:
                raise
        self._onscreen_display = (0, 0, self._width, 30)
        self._hwnd = win32gui.CreateWindow(self._name, self._title,
                                           self._style,
                                           win32con.CW_USEDEFAULT,
                                           win32con.CW_USEDEFAULT,
                                           self._width, self._height,
                                           None, None, self._hinstance, None)
        win32gui.ShowWindow(self._hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self._hwnd)
        # This is used to stop autorepeat which is annoying for our purposes
        self._first_keydown = True
        self._display_string = ""
        self._show_help = False
        self._parameter_changed = False


   # Return the parameter dictionary used to instantiate DomeProjection
    def _parameters(self):
        # setup the parameters dictionary
        image1, image2 = calc_projector_images(self._parameter_values[8],
                                               self._parameter_values[9],
                                               self._parameter_values[6],
                                               self._parameter_values[7])
        parameters = dict(projector_pixel_width = PROJECTOR_PIXEL_WIDTH,
                          projector_pixel_height = PROJECTOR_PIXEL_HEIGHT,
                          first_projector_image = image1,
                          second_projector_image = image2,
                          mirror_radius = self._parameter_values[0],
                          dome_center = [0, self._parameter_values[2],
                                         self._parameter_values[3]],
                          dome_radius = self._parameter_values[1],
                          animal_position = [0, self._parameter_values[4],
                                             self._parameter_values[5]])
        return parameters


    # Save the parameters to a text file
    def saveParameterFile(self, hwnd):
        args = {'hwndOwner' : hwnd,
                'Filter' : "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0",
                'FilterIndex' : 1,
                'DefExt' : "txt",
                'Flags' : win32con.OFN_EXPLORER | win32con.OFN_FILEMUSTEXIST |
                win32con.OFN_HIDEREADONLY}
        file_name = win32gui.GetSaveFileNameW(**args)[0]
        if file_name:
            try:
                parameter_file = open(file_name, "w")
                for i in range(NUM_PARAMETERS):
                    parameter_file.write("%s = %f\n" % (self._parameter_names[i],
                                        self._parameter_values[i]))
                parameter_file.close()
            except:
                win32gui.MessageBox(None, "Error saving parameter file!",
                                "Error", win32con.MB_ICONERROR |
                                win32con.MB_OK)

    
    # Read the parameters from a text file
    def openParameterFile(self, hwnd):
        args = {'hwndOwner' : hwnd,
                'Filter' : "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0",
                'FilterIndex' : 1,
                'DefExt' : "txt",
                'Flags' : win32con.OFN_EXPLORER | win32con.OFN_FILEMUSTEXIST |
                win32con.OFN_HIDEREADONLY}
        file_name = win32gui.GetOpenFileNameW(**args)[0]
        #print file_name
        if file_name:
            try:
                parameter_file = open(file_name, "r")
                for line in parameter_file:
                    parameter_name, parameter_value = line.split(" = ")
                    if parameter_name in self._parameter_names:
                        index = self._parameter_names.index(parameter_name)
                        self._parameter_values[index] = float(parameter_value)
                    else:
                        win32gui.MessageBox(None, "Unknown parameter!",
                                            "Error", win32con.MB_ICONERROR |
                                            win32con.MB_OK)
                parameter_file.close()
            except:
                win32gui.MessageBox(None, "Error reading parameter file!",
                                    "Error", win32con.MB_ICONERROR |
                                    win32con.MB_OK)


    # This function handles all the call backs from the window
    def window_process(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_PAINT:
            self.OnPaint(hwnd, msg, wparam, lparam)

        elif msg == win32con.WM_ERASEBKGND:
            return True

        elif msg == win32con.WM_KEYDOWN:
            self.OnKeyDown(hwnd, msg, wparam, lparam)

        elif msg == win32con.WM_KEYUP:
            self.OnKeyUp(hwnd, msg, wparam, lparam)
        
        elif msg == win32con.WM_DESTROY:
            win32gui.DestroyWindow(hwnd)

        elif msg == win32con.WM_CLOSE:
            win32gui.PostQuitMessage(0)

        elif msg == win32con.WM_ENDSESSION:
            win32gui.PostQuitMessage(0)

        else:
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


    def enterFullscreen(self):
        self._fullscreen = True
        self._style = window_styles["fullscreen"]
        win32gui.SetWindowLong(self._hwnd, win32con.GWL_STYLE, self._style)
        self._width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN) + 1
        self._height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN) + 1
        self._onscreen_display = (0, 0, self._width, 30)
        win32gui.SetWindowPos(self._hwnd, win32con.HWND_TOP, 0, 0,
                              self._width, self._height,
                              win32con.SWP_FRAMECHANGED)


    def exitFullscreen(self):
        self._fullscreen = False
        self._style = window_styles["regular"]
        win32gui.SetWindowLong(self._hwnd, win32con.GWL_STYLE, self._style)
        self._width = PROJECTOR_PIXEL_WIDTH
        self._height = PROJECTOR_PIXEL_HEIGHT
        self._onscreen_display = (0, 0, self._width, 30)
        win32gui.SetWindowPos(self._hwnd, win32con.HWND_TOP, 0, 0,
                              self._width, self._height,
                              win32con.SWP_FRAMECHANGED)


    # Draw the window
    def OnPaint(self, hwnd, msg, wparam, lparam):
        dc, ps = win32gui.BeginPaint(hwnd)
        brush = win32gui.CreateSolidBrush(BLACK)
        win32gui.SelectObject(dc, brush)
        win32gui.Rectangle(dc, 0, 0, PROJECTOR_PIXEL_WIDTH - 1,
                           PROJECTOR_PIXEL_HEIGHT - 1)
        # draw a square of four pixels for each center pixel
        for i in range(len(self._pixels)):
            pixel = self._pixels[i]
            print pixel
            left = int(pixel[1]*self._width/PROJECTOR_PIXEL_WIDTH)
            right = left + 1
            top = int(pixel[0]*self._height/PROJECTOR_PIXEL_HEIGHT)
            bottom = top + 1
            for row in [top, bottom]:
                for col in [left, right]:
                    if (row >= 0 and row < self._height
                        and col >= 0 and col < self._width):
                        #print col, row
                        win32gui.SetPixel(dc, col, row, WHITE)
        win32gui.SetBkColor(dc, BLACK);
        win32gui.SetTextColor(dc, WHITE)
        win32gui.EndPaint(hwnd, ps)
        return 0


    # Determine which key was pressed and display the result of the associated
    # action in the window
    def OnKeyDown(self, hwnd, msg, wparam, lparam):
        if self._first_keydown:
            # only update once, ignore keyboard autorepeat
            self._first_keydown = False
            self._display_string = ""
            if wparam == win32con.VK_UP:
                self._parameter_changed = True
                self._parameter_values[self._current_parameter] += \
                        self._increments[self._current_parameter]
                display_string = ("%s = %.3f" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))
    
            elif wparam == win32con.VK_DOWN:
                self._parameter_changed = True
                self._parameter_values[self._current_parameter] -= \
                        self._increments[self._current_parameter]
                display_string = ("%s = %.3f" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))
    
            elif wparam == win32con.VK_LEFT:
                self._current_parameter = self._current_parameter - 1
                if (self._current_parameter < 0):
                    # roll around to last parameter
                    self._current_parameter = NUM_PARAMETERS - 1
                display_string = ("%s = %.3f" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))
    
            elif wparam == win32con.VK_RIGHT:
                self._current_parameter = self._current_parameter + 1
                self._current_parameter = self._current_parameter % NUM_PARAMETERS
                display_string = ("%s = %.3f" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))
    
            elif wparam == win32con.VK_CONTROL:
                display_string = "increment = %.3f" % \
                        self._increments[self._current_parameter]
    
            elif wparam == win32con.VK_SHIFT:
                display_string = ("%s = %.3f" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))
    
            elif wparam == VK_F:
                if self._fullscreen:
                    self.exitFullscreen()
                else:
                    self.enterFullscreen()

            elif wparam == VK_O:
                # read parameters from a file
                self.openParameterFile(hwnd)
                # update pixels using new parameters and redraw screen
                dome = DomeProjection(**self._parameters())
                self._pixels = \
                        dome.find_projector_pixels(dome.calibration_directions,
                                                   self._pixels)
                win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                                      win32con.RDW_INTERNALPAINT)
    
            elif wparam == VK_S:
                self.saveParameterFile(hwnd)
    
            elif wparam == VK_OEM_PLUS:
                self._increments[self._current_parameter] *= 10.0
                display_string = "increment = %.3f" % \
                        self._increments[self._current_parameter]
    
            elif wparam == VK_OEM_MINUS:
                self._increments[self._current_parameter] /= 10.0
                display_string = "increment = %.3f" % \
                        self._increments[self._current_parameter]
    
            else:
                self._show_help = True
    
            dc = win32gui.GetDC(hwnd)
            brush = win32gui.CreateSolidBrush(BLACK)
            win32gui.SelectObject(dc, brush)
            win32gui.SetTextColor(dc, WHITE)
            win32gui.SetBkColor(dc, BLACK);
            debug = win32gui.DrawText(dc, display_string,
                                      len(display_string),
                                      self._onscreen_display,
                                      win32con.DT_SINGLELINE |
                                      win32con.DT_CENTER |
                                      win32con.DT_VCENTER) 
            win32gui.ReleaseDC(hwnd, dc) 


    # Update the pixel values using the new parameters and redraw the window
    def OnKeyUp(self, hwnd, msg, wparam, lparam):
        # reset _first_keydown for the next keypress
        self._first_keydown = True
        if self._parameter_changed:
            # update the projector pixels using the new parameter values
            dome = DomeProjection(**self._parameters())
            self._pixels = \
                    dome.find_projector_pixels(dome.calibration_directions,
                                               self._pixels)
            self._parameter_changed = False
        # redraw the window to erase onscreen display of the parameter value
        win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                              win32con.RDW_INTERNALPAINT)
        win32gui.InvalidateRect(hwnd, self._onscreen_display, True)

        if self._show_help:
            self._show_help = False
            # show help dialog box
            win32gui.MessageBox(None,
                                "Up Arrow: increase selected parameter\n" +
                                "Down Arrow: decrease selected parameter\n" +
                                "Right Arrow: switch to the next parameter\n" +
                                "Left Arrow: switch to the previous parameter\n" +
                                "Plus: increase parameter increment by 10X\n" +
                                "Minus: decrease parameter increment by 10X\n" +
                                "F: toggle full screen\n" +
                                "O: open pixel file\n" +
                                "S: save pixel file\n" +
                                "Shift: show selected parameter\n" +
                                "Control: show selected parameter increment\n",
                                "Help", win32con.MB_ICONINFORMATION |
                                win32con.MB_OK)


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main_window = MainWindow()
    # run until PostQuitMessage()
    win32gui.PumpMessages()


