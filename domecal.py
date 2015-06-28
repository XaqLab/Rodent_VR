import winxpgui as win32gui
import win32api
import win32con

NUM_PARAMETERS = 2
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Color constants
BACKGROUND = win32api.RGB(0, 0, 0)
FOREGROUND = win32api.RGB(255, 255, 255)
FOREGROUND = win32api.RGB(0, 255, 0)

VK_OEM_PLUS = 0xbb
VK_OEM_MINUS = 0xbd
VK_O = 0x4f
VK_S = 0x53

class MainWindow():
    """ The main application window class """

    def __init__(self):
        self._current_parameter = 0
        self._parameter_values = [360, 640]
        self._parameter_names = ["Row", "Column"]
        self._increment = 1.0

        self._name = "MainWindow"
        self._title = "Dome Calibration"
        self._width = IMAGE_WIDTH
        self._height = IMAGE_HEIGHT
        self._style = (win32con.WS_THICKFRAME | win32con.WS_VISIBLE |
                       win32con.WS_SYSMENU | win32con.DS_SETFONT |
                       win32con.WS_MINIMIZEBOX)

        win32gui.InitCommonControls()
        self._hinstance = win32gui.dllhandle
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = self._name
        wc.style =  win32con.CS_GLOBALCLASS|win32con.CS_VREDRAW | win32con.CS_HREDRAW
        wc.hCursor = win32gui.LoadCursor( 0, win32con.IDC_ARROW )
        wc.lpfnWndProc = self.wndproc

        try:
            classAtom = win32gui.RegisterClass(wc)
        except win32gui.error, err_info:
            if err_info.winerror!=winerror.ERROR_CLASS_ALREADY_EXISTS:
                raise

        self._onscreen_display = (0, 0, IMAGE_WIDTH, 30)
        self._hwnd = win32gui.CreateWindow(self._name, self._title,
                                           self._style,
                                           win32con.CW_USEDEFAULT,
                                           win32con.CW_USEDEFAULT,
                                           self._width, self._height,
                                           None, None, self._hinstance, None)
        win32gui.ShowWindow(self._hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self._hwnd)


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
                    parameter_file.write("%s = %d\n" % (self._parameter_names[i],
                                        self._parameter_values[i]))
                parameter_file.close()
            except:
                win32gui.MessageBox(None, "Error saving parameter file!",
                                "Error", win32con.MB_ICONERROR |
                                win32con.MB_OK)

    
    # Read the parameters from a text file
    def openParameterFile(self, hwnd):
                #'Filter' : 'Text Files (*.txt)\\0*.txt\\0All Files (*.*)\\0*.*\\0',
        args = {'hwndOwner' : hwnd,
                'Filter' : "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0",
                'FilterIndex' : 1,
                'DefExt' : "txt",
                'Flags' : win32con.OFN_EXPLORER | win32con.OFN_FILEMUSTEXIST |
                win32con.OFN_HIDEREADONLY}
        file_name = win32gui.GetOpenFileNameW(**args)[0]
        print file_name
        if file_name:
            print "file_name is true"
            try:
                parameter_file = open(file_name, "r")
                for line in parameter_file:
                    parameter_name, equal_sign, parameter_value = line.split()
                    print parameter_name
                    print parameter_value
                    if parameter_name in self._parameter_names:
                        index = self._parameter_names.index(parameter_name)
                        self._parameter_values[index] = int(parameter_value)
                    else:
                        win32gui.MessageBox(None, "Unknown parameter!",
                                            "Error", win32con.MB_ICONERROR |
                                            win32con.MB_OK)
                parameter_file.close()
            except:
                win32gui.MessageBox(None, "Error reading parameter file!",
                                    "Error", win32con.MB_ICONERROR |
                                    win32con.MB_OK)


    # wndproc
    # messages = {15: "WM_PAINT", 133: "WM_NCPAINT"}
    def wndproc(self, hwnd, msg, wparam, lparam):
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

        #elif msg == win32con.WM_SYSCOMMAND:
        #    if wparam == win32con.SC_CLOSE:
        #        win32gui.PostQuitMessage(0)

        elif msg == win32con.WM_CLOSE:
            win32gui.PostQuitMessage(0)

        elif msg == win32con.WM_ENDSESSION:
            win32gui.PostQuitMessage(0)

        else:
            #messages_to_ignore = [6, 8, 28, 31, 32, 33, 132, 133, 134, 512, 645]
            #if msg not in messages_to_ignore:
                #print "unhandeled message:", msg, wparam, lparam
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


    def OnPaint(self, hwnd, msg, wparam, lparam):
        dc, ps = win32gui.BeginPaint(hwnd)
        brush = win32gui.CreateSolidBrush(BACKGROUND)
        win32gui.SelectObject(dc, brush)
        win32gui.Rectangle(dc, 0, 0, IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1)
        #win32gui.ExtFloodFill(dc, IMAGE_WIDTH/2, IMAGE_HEIGHT/2, BACKGROUND,
                          #    win32con.FLOODFILLBORDER)
        win32gui.SetPixel(dc,
                          int(self._parameter_values[1]),
                          int(self._parameter_values[0]),
                          FOREGROUND)
        win32gui.SetBkColor(dc, BACKGROUND);
        win32gui.SetTextColor(dc, FOREGROUND)
        win32gui.EndPaint(hwnd, ps)
        return 0


    def OnKeyDown(self, hwnd, msg, wparam, lparam):
        display_string = ""
        if wparam == win32con.VK_UP:
            self._parameter_values[self._current_parameter] += self._increment
            display_string = ("%s = %d" %
                              (self._parameter_names[self._current_parameter],
                               self._parameter_values[self._current_parameter]))

        elif wparam == win32con.VK_DOWN:
            self._parameter_values[self._current_parameter] -= self._increment
            display_string = ("%s = %d" %
                              (self._parameter_names[self._current_parameter],
                               self._parameter_values[self._current_parameter]))

        elif wparam == win32con.VK_LEFT:
            self._current_parameter = self._current_parameter - 1
            if (self._current_parameter < 0):
                # roll around to last parameter
                self._current_parameter = NUM_PARAMETERS - 1
                display_string = ("%s = %d" %
                                  (self._parameter_names[self._current_parameter],
                                   self._parameter_values[self._current_parameter]))

        elif wparam == win32con.VK_RIGHT:
            self._current_parameter = self._current_parameter + 1
            self._current_parameter = self._current_parameter % NUM_PARAMETERS
            display_string = ("%s = %d" %
                              (self._parameter_names[self._current_parameter],
                               self._parameter_values[self._current_parameter]))

        elif wparam == win32con.VK_CONTROL:
            display_string = "increment = %f" % self._increment

        elif wparam == win32con.VK_SHIFT:
            display_string = ("%s = %d" %
                              (self._parameter_names[self._current_parameter],
                               self._parameter_values[self._current_parameter]))

        elif wparam == VK_O:
            self.openParameterFile(hwnd)
            win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                                  win32con.RDW_INTERNALPAINT)

        elif wparam == VK_S:
            self.saveParameterFile(hwnd)

        elif wparam == VK_OEM_PLUS:
            self._increment = 10.0 * self._increment
            display_string = "increment = %f" % self._increment

        elif wparam == VK_OEM_MINUS:
            self._increment = self._increment / 10.0
            display_string = "increment = %f" % self._increment

        else:
            print "keypress: %x" % wparam

        dc = win32gui.GetDC(hwnd)
        brush = win32gui.CreateSolidBrush(BACKGROUND)
        win32gui.SelectObject(dc, brush)
        win32gui.SetTextColor(dc, FOREGROUND)
        win32gui.SetBkColor(dc, BACKGROUND);
        debug = win32gui.DrawText(dc, display_string,
                                  len(display_string),
                                  self._onscreen_display,
                                  win32con.DT_SINGLELINE |
                                  win32con.DT_CENTER |
                                  win32con.DT_VCENTER) 
        win32gui.ReleaseDC(hwnd, dc) 


    def OnKeyUp(self, hwnd, msg, wparam, lparam):
        # This is where the minimization will update the projector pixels using
        # the new parameter values
        win32gui.RedrawWindow(hwnd, None, None, win32con.RDW_INVALIDATE |
                              win32con.RDW_INTERNALPAINT)
        win32gui.InvalidateRect(hwnd, self._onscreen_display, True)


    """
    def OnSize(self, hwnd, msg, wparam, lparam):
        x = win32api.LOWORD(lparam)
        y = win32api.HIWORD(lparam)
        self._DoSize(x,y)
        return 1
    """


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main_window = MainWindow()
    # run until PostQuitMessage()
    win32gui.PumpMessages()


