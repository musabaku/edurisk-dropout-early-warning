"""SHAP explanations: why the model thinks a student is at risk.

TreeSHAP (Lundberg et al., 2020) gives each feature a signed contribution to the
predicted Dropout score; contributions add up to the model output for that student.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config as C


def dropout_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values for the Dropout class (n_students x n_features) and the base value."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):                 # older SHAP: one array per class
        drop = values[C.DROPOUT]
    else:                                         # newer SHAP: (n, features, classes)
        drop = values[:, :, C.DROPOUT]
    base = np.atleast_1d(explainer.expected_value)[C.DROPOUT]
    return np.asarray(drop), float(base)


def global_importance(shap_drop: np.ndarray, columns: list[str]) -> pd.DataFrame:
    imp = np.abs(shap_drop).mean(axis=0)
    return (pd.DataFrame({"feature": columns, "mean_abs_shap": imp})
              .assign(label=lambda d: d["feature"].map(C.PRETTY).fillna(d["feature"]))
              .sort_values("mean_abs_shap", ascending=False, ignore_index=True))


def describe_value(feature: str, value: float) -> str:
    """Turn a raw feature value into words a teacher understands."""
    yes_no = {"debtor", "tuition_up_to_date", "scholarship_holder", "displaced",
              "special_needs", "international", "daytime_attendance"}
    if feature in yes_no:
        return "yes" if value >= 0.5 else "no"
    if feature == "gender":
        return "male" if value >= 0.5 else "female"
    if feature == "course":
        return C.COURSE_NAMES.get(int(value), str(int(value)))
    if feature.endswith("approval_rate"):
        return f"{value:.0%}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def top_reasons(shap_row: np.ndarray, x_row: pd.Series, k: int = 3) -> list[dict]:
    """The k features that push this student's Dropout risk UP the most."""
    order = np.argsort(-shap_row)
    reasons = []
    for idx in order[:k]:
        if shap_row[idx] <= 0:
            break
        feat = x_row.index[idx]
        reasons.append({
            "feature": feat,
            "text": f"{C.PRETTY.get(feat, feat)} = {describe_value(feat, x_row.iloc[idx])}",
            "impact": float(shap_row[idx]),
        })
    return reasons
