"""Feature construction per checkpoint.

A checkpoint is a moment in the first year when the school re-scores every student:
at enrollment (no marks yet), after semester 1, after semester 2. Each later checkpoint
uses everything the earlier one had plus the new marks. Nothing from the future leaks in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den, with 0 where den is 0 (student enrolled in no units that term)."""
    return pd.Series(np.where(den > 0, num / den.where(den > 0, 1), 0.0), index=num.index)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add pass rates and grade trend. Returns a copy."""
    out = df.copy()
    out["sem1_approval_rate"] = _safe_ratio(out["sem1_approved"], out["sem1_enrolled"])
    out["sem2_approval_rate"] = _safe_ratio(out["sem2_approved"], out["sem2_enrolled"])
    out["grade_trend"] = out["sem2_grade"] - out["sem1_grade"]
    return out


def feature_list(checkpoint: str) -> list[str]:
    if checkpoint not in C.CHECKPOINTS:
        raise KeyError(f"Unknown checkpoint '{checkpoint}'. Use one of {list(C.CHECKPOINTS)}")
    return list(C.CHECKPOINTS[checkpoint])


def make_xy(df: pd.DataFrame, checkpoint: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Feature matrix X for the checkpoint, and encoded target y if the label column exists."""
    full = add_engineered(df)
    X = full[feature_list(checkpoint)].astype(float)
    y = None
    if C.TARGET in full.columns:
        from .data import encode_target
        y = encode_target(full[C.TARGET])
    return X, y


def nominal_in(columns: list[str]) -> list[str]:
    return [c for c in columns if c in C.NOMINAL]
