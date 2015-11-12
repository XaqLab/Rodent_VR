# This Makefile will build the PyInstaller Win32 application.

all: caldev.exe

caldev.exe: caldev.py
	export WINEPREFIX=~/.wine-dev; wine pyinstaller --additional-hooks-dir=./hooks --onefile caldev.py

clean:
	rm -f caldev.spec
	rm -rf build
	rm -rf dist

