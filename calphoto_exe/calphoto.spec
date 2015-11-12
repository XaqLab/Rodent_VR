# -*- mode: python -*-
a = Analysis(['calphoto.py'],
             pathex=['Z:\\Users\\jwbwater\\BCM\\FireflyProject\\DomeProjection\\Python\\calphoto_exe'],
             hiddenimports=[],
             hookspath=['./hooks'],
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='calphoto.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
