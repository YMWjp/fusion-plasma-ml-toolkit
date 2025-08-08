# src/utils/paths.py
from pathlib import Path

# src directory
SRC_DIR: Path = Path(__file__).resolve().parent.parent
# data root
DATA_DIR: Path = SRC_DIR / "data"
# each sub directory
DATASETS_DIR: Path = DATA_DIR / "datasets"
LOGS_DIR: Path = DATA_DIR / "logs"

EGDATA_DIR: Path = DATA_DIR / "egdata"
SDL_LOOP_DATA_DIR: Path = DATA_DIR / "SDLloopdata"

EXPERIMENT_LOG_CSV: Path = DATA_DIR / "experiment_log_new.csv"

# create directories if not exist
for _p in (DATA_DIR, DATASETS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)