# src/utils/utils.py
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Sequence, Iterable
import csv
import numpy as np
from src.utils.paths import DATASETS_DIR, LOGS_DIR

def write_csv_header(filepath: Path | str, header: Sequence[str],
                     delimiter: str = ',', overwrite: bool = False) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(list(header))

def get_dataset_path(filename: str) -> Path:
    return DATASETS_DIR / filename

def log_error_shot(shot_no: int, log_filename: str = "errorshot.txt") -> Path:
    log_path = LOGS_DIR / log_filename
    now = datetime.now(ZoneInfo("Asia/Tokyo"))

    # append date and shot number to log file
    with log_path.open(mode='a', encoding='utf-8', newline='') as f:
        f.write(f"{now:%Y-%m-%d}  shotNO:{shot_no}\n")
    return log_path

def append_rows_to_csv(filepath: Path | str, rows: np.ndarray | Iterable[Iterable], *,
                       delimiter: str = ',', floatfmt: str = '%.10e') -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='') as f:
        np.savetxt(f, rows, delimiter=delimiter, fmt=floatfmt)