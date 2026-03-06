from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def list_csv_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("**/*.csv"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
