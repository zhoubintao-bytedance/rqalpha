"""Frozen AX1 research replay helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fold_frame(artifact_root: str | Path, ref: str) -> pd.DataFrame:
    path = Path(artifact_root) / ref
    if not path.is_file():
        raise FileNotFoundError(f"missing fold artifact: {path}")
    return pd.read_parquet(path)

