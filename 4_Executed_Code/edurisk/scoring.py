"""Score new students with a saved model - used by the dashboard's upload page.

Uploads are untrusted: read_any() and prepare() turn every bad file into a ValueError with a
plain message (or a list of skipped rows) instead of letting pandas/numpy exceptions escape.
"""
from __future__ import annotations

import io

import joblib
import pandas as pd

from . import config as C
from . import evaluate, explain, features


def load_bundle(checkpoint: str) -> dict:
    """{'model', 'features', 'name', 'checkpoint'} saved by run_pipeline.py"""
    return joblib.load(C.MODEL_DIR / f"edurisk_{checkpoint}.joblib")


def read_any(file_bytes: bytes) -> pd.DataFrame:
    """Accept the original UCI layout (';' separated, long headers) or clean names (',')."""
    if not file_bytes or not file_bytes.strip():
        raise ValueError("The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:                       # csv.Error, ParserError, UnicodeDecodeError
        raise ValueError(f"Could not read the file as CSV ({exc}).") from exc
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=C.RAW_TO_CLEAN)
    if "student_id" not in df.columns:
        df.insert(0, "student_id", range(1, len(df) + 1))
    return df


def required_columns(checkpoint: str) -> list[str]:
    """Raw columns this checkpoint's model needs (engineered ones are computed here)."""
    return [f for f in features.feature_list(checkpoint)
            if f not in C.SEM1_ENGINEERED + C.SEM2_ENGINEERED]


def missing_columns(df: pd.DataFrame, checkpoint: str) -> list[str]:
    return [c for c in required_columns(checkpoint) if c not in df.columns]


def prepare(df: pd.DataFrame, checkpoint: str) -> tuple[pd.DataFrame, list[int]]:
    """Validate an upload. Returns (rows that can be scored, student_ids of skipped rows).

    Raises ValueError when nothing can be scored (no rows, missing columns, no valid row).
    A row is skipped when a required value is blank or not a number.
    """
    if df.empty:
        raise ValueError("The file has a header but no student rows.")
    missing = missing_columns(df, checkpoint)
    if missing:
        raise ValueError(f"Missing columns for this checkpoint: {missing}")
    out = df.copy()
    req = required_columns(checkpoint)
    out[req] = out[req].apply(pd.to_numeric, errors="coerce")
    bad = out[req].isna().any(axis=1)
    skipped = out.loc[bad, "student_id"].tolist()
    out = out.loc[~bad].reset_index(drop=True)
    if out.empty:
        raise ValueError("No row could be scored: every row has a blank or non-numeric value "
                         "in a required column.")
    return out, skipped


def score(df: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    """Probabilities, risk tier and top-3 reasons for every row of a validated dataframe."""
    bundle = load_bundle(checkpoint)
    frame = df.copy()
    for col in ["sem2_approved", "sem2_enrolled", "sem2_grade", "sem1_approved",
                "sem1_enrolled", "sem1_grade"]:
        if col not in frame.columns:
            frame[col] = 0          # only feeds engineered columns this checkpoint ignores
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    X, _ = features.make_xy(frame.drop(columns=[C.TARGET], errors="ignore"), checkpoint)
    X = X[bundle["features"]]
    proba = bundle["model"].predict_proba(X)
    shap_drop, _ = explain.dropout_shap(bundle["model"], X)
    out = pd.DataFrame({
        "student_id": frame["student_id"].to_numpy(),
        "p_dropout": proba[:, 0], "p_enrolled": proba[:, 1], "p_graduate": proba[:, 2],
        "risk_tier": evaluate.risk_tier(proba[:, 0]),
    })
    out["top_reasons"] = [" | ".join(r["text"] for r in explain.top_reasons(shap_drop[i], X.iloc[i]))
                          for i in range(len(X))]
    return out.sort_values("p_dropout", ascending=False, ignore_index=True)
