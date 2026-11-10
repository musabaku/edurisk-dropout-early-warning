"""Group-level error check (Baker & Hawn, 2022): does the warning work equally well for
every group of students, or does it miss more dropouts in one group?

'Flagged' means the student landed in the High risk tier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import risk_tier


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 20, 25, 200], labels=["17-20", "21-25", "26+"]).astype(str)


GROUPS = {
    "Gender": lambda d: d["gender"].map({0: "Female", 1: "Male"}),
    "Scholarship": lambda d: d["scholarship_holder"].map({0: "No scholarship", 1: "Scholarship"}),
    "Age at enrollment": lambda d: _age_band(d["age_at_enrollment"]),
    "Displaced": lambda d: d["displaced"].map({0: "Lives at home", 1: "Displaced"}),
}


def group_report(df: pd.DataFrame, y_true, p_dropout) -> pd.DataFrame:
    y_drop = np.asarray(y_true) == C.DROPOUT
    flagged = risk_tier(p_dropout) == "High"
    rows = []
    for attribute, make_group in GROUPS.items():
        groups = make_group(df).to_numpy()
        for g in pd.unique(groups):
            m = groups == g
            pos, neg = y_drop[m], ~y_drop[m]
            rows.append({
                "attribute": attribute,
                "group": g,
                "students": int(m.sum()),
                "actual_dropout_rate": float(pos.mean()),
                "flag_rate": float(flagged[m].mean()),
                "recall": float(flagged[m][pos].mean()) if pos.any() else np.nan,
                "false_positive_rate": float(flagged[m][neg].mean()) if neg.any() else np.nan,
            })
    return pd.DataFrame(rows)


def recall_gap(report: pd.DataFrame) -> pd.DataFrame:
    """Largest recall difference inside each attribute (smaller is fairer)."""
    return (report.groupby("attribute")["recall"].agg(lambda s: s.max() - s.min())
                  .rename("recall_gap").reset_index())
