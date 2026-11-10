"""Metrics, cross-validation and risk tiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import models as M


def metrics(y_true, proba: np.ndarray) -> dict:
    """All scores used in the report. Dropout is treated as the positive class."""
    y_true = np.asarray(y_true)
    pred = proba.argmax(axis=1)
    is_drop, pred_drop = y_true == C.DROPOUT, pred == C.DROPOUT
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "dropout_precision": precision_score(is_drop, pred_drop, zero_division=0),
        "dropout_recall": recall_score(is_drop, pred_drop, zero_division=0),
        "dropout_f1": f1_score(is_drop, pred_drop, zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "dropout_auc": roc_auc_score(is_drop, proba[:, C.DROPOUT]),
    }


def cross_validate(name: str, X: pd.DataFrame, y: pd.Series,
                   xgb_params: dict | None = None) -> dict:
    """5-fold stratified CV on the training set. Returns mean and std of every metric."""
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    rows = []
    for tr, va in skf.split(X, y):
        model = M.fit(M.build(name, list(X.columns), xgb_params), name,
                      X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[va], model.predict_proba(X.iloc[va])))
    df = pd.DataFrame(rows)
    return {"mean": df.mean().to_dict(), "std": df.std(ddof=1).to_dict()}


def confusion(y_true, proba) -> np.ndarray:
    return confusion_matrix(y_true, proba.argmax(axis=1), labels=[0, 1, 2])


def risk_tier(p_dropout) -> np.ndarray:
    p = np.asarray(p_dropout, dtype=float)
    return np.where(p >= C.HIGH_RISK, "High", np.where(p >= C.MEDIUM_RISK, "Medium", "Low"))


def tier_table(y_true, p_dropout) -> pd.DataFrame:
    """How many students land in each tier and how many of them really dropped out."""
    df = pd.DataFrame({"tier": risk_tier(p_dropout),
                       "dropout": np.asarray(y_true) == C.DROPOUT})
    total_drop = df["dropout"].sum()
    out = (df.groupby("tier")["dropout"].agg(students="size", actual_dropouts="sum")
             .reindex(["High", "Medium", "Low"]).fillna(0).astype(int).reset_index())
    out["share_of_students"] = out["students"] / len(df)
    out["dropout_rate_in_tier"] = out["actual_dropouts"] / out["students"].replace(0, np.nan)
    out["share_of_all_dropouts"] = out["actual_dropouts"] / total_drop
    return out
