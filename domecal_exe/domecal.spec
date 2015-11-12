# -*- mode: python -*-
a = Analysis(['domecal.py'],
             pathex=['Z:\\Users\\jwbwater\\BCM\\FireflyProject\\DomeProjection\\Python\\domecal_exe'],
             hiddenimports=[],
             hookspath=['./hooks'],
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='domecal.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
