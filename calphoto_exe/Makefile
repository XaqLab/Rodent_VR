# This Makefile will build the PyInstaller Win32 application.

all: calphoto.exe

calphoto.exe: calphoto.py
	export WINEPREFIX=~/.wine-dev; wine pyinstaller --additional-hooks-dir=./hooks --onefile calphoto.py

clean:
	rm -f calphoto.spec
	rm -rf build
	rm -rf dist

