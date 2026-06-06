from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_col(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[\s\-/\\:]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return re.sub(r"_+", "_", name).strip("_")


def normalize_columns(columns) -> dict[str, str]:
    return {col: normalize_col(col) for col in columns}


def find_first(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    col_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        hit = col_lower.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def build_column_map(
    columns: list[str], mapping: dict[str, tuple[str, ...]]
) -> dict[str, str | None]:
    return {std: find_first(columns, cands) for std, cands in mapping.items()}


def list_csv_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.csv"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
