This bare bones MS Windows GUI was developed on a Mac using wine.

The development wine bottle, wine-dev, has python installed and uses the python
library libpython27.a to build a Windows executable that does not require a python installation.

To use the development wine bottle which has python and the python libraries
required to build this application execute this instruction at the command line:

export WINEPREFIX=~/.wine-dev

To build caldev.exe simply type "make" at the command line in this directory. When the build is finished caldev.exe will be found in the dist directory.
