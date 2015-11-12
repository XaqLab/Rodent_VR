# import stuff from numpy
from numpy import pi, sin, cos

# import win32 stuff
import winxpgui as win32gui
import win32api
import win32con

# import dome projection stuff
from dome_projection import DomeProjection

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
pixel_colors = [WHITE, GREEN, WHITE,
                CYAN, YELLOW, CYAN,
                WHITE, GREEN, WHITE]

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
VK_N = 0x4e
VK_O = 0x4f
VK_P = 0x50
VK_Q = 0x51
VK_S = 0x53
VK_OEM_PLUS = 0xbb
VK_OEM_MINUS = 0xbd
VK_ESCAPE = 0x1b

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
        # initial projector pixel values
        pixels = [[341.5, 640.5], [353.5, 764.5], [378.5, 743.5],
                  [395.5, 699.5], [402.5, 640.5], [395.5, 582.5],
                  [379.5, 539.5], [356.5, 517.5], [402.5, 903.5],
                  [448.5, 853.5], [478.5, 757.5], [489.5, 639.5],
                  [480.5, 525.5], [452.5, 429.5], [404.5, 378.5],
                  [509.5, 1060.5], [566.5, 962.5], [594.5, 808.5],
                  [603.5, 639.5], [595.5, 471.5], [565.5, 325.5],
                  [514.5, 231.5]]
        points = [[p[1], p[0]] for p in pixels]
        # convert (u,v) to (row, column)
        self._pixels = [[int(p[1]) + 0.5, int(p[0]) + 0.5] for p in points]
        # initialize the increment for each pixel in the image
        self._increment = 10
        # initialize the pixel selected for moving
        self._selected_pixel = 0
        # setup the window
        self._name = "MainWindow"
        self._title = "Taking Photos for Dome Calibration"
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


    # Save the parameters to a text file
    def saveParameterFile(self, hwnd):
        args = {'hwndOwner' : hwnd,
                'Filter' : "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0",
                'FilterIndex' : 1,
                'DefExt' : "txt",
                'Flags' : win32con.OFN_EXPLORER | win32con.OFN_FILEMUSTEXIST |
                win32con.OFN_HIDEREADONLY}
        try:
            file_name = win32gui.GetSaveFileNameW(**args)[0]
            if file_name:
                try:
                    parameter_file = open(file_name, "w")
                    for pixel in self._pixels:
                        parameter_file.write("%.1f, %.1f\n" % (pixel[0], pixel[1]))
                    parameter_file.close()
                except Exception, e:
                    print str(e)
                    win32gui.MessageBox(None, "Error saving parameter file!",
                                    "Error", win32con.MB_ICONERROR |
                                    win32con.MB_OK)
        except:
            # no file selected
            pass

    
    # Read the parameters from a text file
    def openParameterFile(self, hwnd):
        args = {'hwndOwner' : hwnd,
                'Filter' : "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0",
                'FilterIndex' : 1,
                'DefExt' : "txt",
                'Flags' : win32con.OFN_EXPLORER | win32con.OFN_FILEMUSTEXIST |
                win32con.OFN_HIDEREADONLY}
        try:
            file_name = win32gui.GetOpenFileNameW(**args)[0]
            if file_name:
                try:
                    parameter_file = open(file_name, "r")
                    for i in range(len(self._pixels)):
                        line = parameter_file.readline()
                        row, col = line.split(", ")
                        self._pixels[i] = [float(row), float(col)]
                    parameter_file.close()
                except Exception, e:
                    print str(e)
                    win32gui.MessageBox(None, "Error reading parameter file!",
                                        "Error", win32con.MB_ICONERROR |
                                        win32con.MB_OK)
        except:
            # no file selected
            pass


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
        win32gui.Rectangle(dc, 0, 0, self._width - 1, self._height - 1)
        # draw a square of four pixels for each center pixel
        for i in range(len(self._pixels)):
            pixel = self._pixels[i]
            left = int(pixel[1]*self._width/PROJECTOR_PIXEL_WIDTH)
            right = left + 1
            top = int(pixel[0]*self._height/PROJECTOR_PIXEL_HEIGHT)
            bottom = top + 1
            for row in [top, bottom]:
                for col in [left, right]:
                    #win32gui.SetPixel(dc, col, row, pixel_colors[i])
                    win32gui.SetPixel(dc, col, row, GREEN)
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
                self._pixels[self._selected_pixel][0] -= self._increment
    
            elif wparam == win32con.VK_DOWN:
                self._pixels[self._selected_pixel][0] += self._increment
    
            elif wparam == win32con.VK_LEFT:
                self._pixels[self._selected_pixel][1] -= self._increment
    
            elif wparam == win32con.VK_RIGHT:
                self._pixels[self._selected_pixel][1] += self._increment
    
            elif wparam == VK_1:
                self._selected_pixel = 0
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_2:
                self._selected_pixel = 1
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_3:
                self._selected_pixel = 2
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_4:
                self._selected_pixel = 3
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_5:
                self._selected_pixel = 4
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_6:
                self._selected_pixel = 5
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_7:
                self._selected_pixel = 6
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_8:
                self._selected_pixel = 7
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_9:
                self._selected_pixel = 8
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == win32con.VK_CONTROL:
                self._display_string = "increment = %d" % self._increment
    
            elif wparam == win32con.VK_SHIFT:
                self._display_string = "Pixel " + str(self._selected_pixel + 1)
    
            elif wparam == VK_F:
                if self._fullscreen:
                    self.exitFullscreen()
                else:
                    self.enterFullscreen()

            elif wparam == VK_N:
                if self._selected_pixel == len(self._pixels) - 1:
                    self._selected_pixel = 0
                else:
                    self._selected_pixel += 1
                self._display_string = "Pixel " + str(self._selected_pixel + 1)

            elif wparam == VK_O:
                # read parameters and pixels from a file
                self.openParameterFile(hwnd)
                # redraw the screen using the new pixels
                win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                                      win32con.RDW_INTERNALPAINT)
    
            elif wparam == VK_P:
                if self._selected_pixel == 0:
                    self._selected_pixel = len(self._pixels) - 1
                else:
                    self._selected_pixel -= 1
                self._display_string = "Pixel " + str(self._selected_pixel + 1)

            elif wparam == VK_Q:
                win32gui.PostQuitMessage(0)

            elif wparam == VK_S:
                self.saveParameterFile(hwnd)
    
            elif wparam == VK_ESCAPE:
                self.exitFullscreen()

            elif wparam == VK_OEM_PLUS:
                self._increment = 10
                self._display_string = "increment = %d" % self._increment
    
            elif wparam == VK_OEM_MINUS:
                self._increment = 1
                self._display_string = "increment = %d" % self._increment
    
            else:
                self._show_help = True
    
            dc = win32gui.GetDC(hwnd)
            brush = win32gui.CreateSolidBrush(BLACK)
            win32gui.SelectObject(dc, brush)
            win32gui.SetTextColor(dc, WHITE)
            win32gui.SetBkColor(dc, BLACK);
            debug = win32gui.DrawText(dc, self._display_string,
                                      len(self._display_string),
                                      self._onscreen_display,
                                      win32con.DT_SINGLELINE |
                                      win32con.DT_CENTER |
                                      win32con.DT_VCENTER) 
            win32gui.ReleaseDC(hwnd, dc) 


    # Update the pixel values using the new parameters and redraw the window
    def OnKeyUp(self, hwnd, msg, wparam, lparam):
        # reset _first_keydown for the next keypress
        self._first_keydown = True
        # redraw the window
        win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                              win32con.RDW_INTERNALPAINT)
        if self._display_string:
            # invalidate the onscreen display
            win32gui.InvalidateRect(hwnd, self._onscreen_display, True)
        if self._show_help:
            self._show_help = False
            # show help dialog box
            win32gui.MessageBox(None,
                                "Up Arrow: move selected pixel up\n" +
                                "Down Arrow: move selected pixel down\n" +
                                "Right Arrow: move selected pixel right\n" +
                                "Left Arrow: move selected pixel left\n" +
                                "Plus: change pixel increment to 10 pixels\n" +
                                "Minus: change pixel increment to 1 pixel\n" +
                                "N: select next pixel\n" +
                                "P: select previous pixel\n" +
                                "F: toggle full screen\n" +
                                "O: open pixel file\n" +
                                "S: save pixel file\n" +
                                "Q: quit\n" +
                                "Shift: show select pixel number\n" +
                                "Control: show current pixel increment\n",
                                "Help", win32con.MB_ICONINFORMATION |
                                win32con.MB_OK)


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main_window = MainWindow()
    # run until PostQuitMessage()
    win32gui.PumpMessages()


