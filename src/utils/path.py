# src/utils/paths.py
from pathlib import Path

# src directory
SRC_DIR: Path = Path(__file__).resolve().parent.parent
# data root
DATA_DIR: Path = SRC_DIR / "data"
# each sub directory
DATASETS_DIR: Path = DATA_DIR / "datasets"
LOGS_DIR: Path = DATA_DIR / "logs"

# create directories if not exist
for _p in (DATA_DIR, DATASETS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)