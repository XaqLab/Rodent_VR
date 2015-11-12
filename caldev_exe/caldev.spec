# -*- mode: python -*-
a = Analysis(['caldev.py'],
             pathex=['Z:\\Users\\jwbwater\\BCM\\FireflyProject\\DomeProjection\\Python\\caldev_exe'],
             hiddenimports=[],
             hookspath=['./hooks'],
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='caldev.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
