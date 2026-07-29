
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("sklearn")
    + collect_submodules("cma")
    + collect_submodules("gplearn")
    + collect_submodules("torch")
    + ["matplotlib.backends.backend_qtagg"]
)

a = Analysis(
    ["entrypoint/gui.py"],
    pathex=["."],
    binaries=[],
    datas=[("config/data.xlsx", "config")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="beam_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
