# This Makefile builds the PyInstaller Win32 application on a Mac using wine.
# The wine bottle wine-dev has python installed because it is used to build
# caldev.exe using a python package called pyinstaller. Another wine bottle
# that does not have python installed is used to test caldev.exe to make sure
# it can run on an MS Windows machine that does not have python installed.

all: caldev.exe

caldev.exe: caldev.py
	export WINEPREFIX=~/.wine-dev; wine pyinstaller --additional-hooks-dir=./hooks --onefile caldev.py

clean:
	rm -f caldev.spec
	rm -rf build
	rm -rf dist

