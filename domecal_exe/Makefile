# This Makefile will build the PyInstaller Win32 application.

all: domecal.exe

domecal.exe: domecal.py
	export WINEPREFIX=~/.wine-dev; wine pyinstaller --additional-hooks-dir=./hooks --onefile domecal.py

clean:
	rm -f domecal.spec
	rm -rf build
	rm -rf dist

