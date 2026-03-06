from __future__ import annotations

from typing import Dict, Iterable, Optional


def normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    normalized = {}
    for col in columns:
        key = col.strip().lower().replace(" ", "_").replace("-", "_")
        normalized[col] = key
    return normalized


def find_first(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def build_column_map(columns: Iterable[str], mapping: Dict[str, Iterable[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for standard, candidates in mapping.items():
        match = find_first(columns, candidates)
        if match is not None:
            result[standard] = match
    return result
