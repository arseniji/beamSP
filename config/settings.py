import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    ROOT = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).resolve().parent
    RESULTS_DIR = EXE_DIR / "results"
    _external_data = EXE_DIR / "data.xlsx"
    DATA_PATH = _external_data if _external_data.exists() else ROOT / "config" / "data.xlsx"
else:
    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"
    DATA_PATH = ROOT / "config" / "data.xlsx"

SEED = 1337
FEATURES = ["a_h0", "is_steel", "H", "s", "R", "E"]
TARGETS = {
    "SP63": "Qдв_СП63",
    "RUK78": "Qдв_рук78",
}

SYNTH = {
    "enabled": True,
    "samples_per_profile": 15,
    "noise": {
        "H": 3.0,
        "s": 0.3,
        "R": 3.0,
        "E": 50.0,
    },
}
ACTIVE_MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "bio_ga",
]
