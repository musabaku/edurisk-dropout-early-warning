"""Loading, cleaning and splitting the student dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw(path: Path | str = C.DATA_FILE) -> pd.DataFrame:
    """Read the UCI CSV (semicolon separated, UTF-8 with BOM) and give columns clean names."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]          # one header has a trailing tab
    missing = set(C.RAW_TO_CLEAN) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file layout, missing columns: {sorted(missing)}")
    df = df.rename(columns=C.RAW_TO_CLEAN)
    df.insert(0, "student_id", range(1, len(df) + 1))    # stable id for the database/dashboard
    return df


def validate(df: pd.DataFrame) -> dict:
    """Basic data-quality checks; raises if the data is unusable, returns a summary otherwise."""
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    bad = set(df[C.TARGET].unique()) - set(C.CLASSES)
    if bad:
        raise ValueError(f"Unknown target labels: {bad}")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.drop(columns="student_id").duplicated().sum()),
        "class_counts": df[C.TARGET].value_counts().reindex(C.CLASSES).astype(int).to_dict(),
    }


def encode_target(y: pd.Series) -> pd.Series:
    """Dropout -> 0, Enrolled -> 1, Graduate -> 2."""
    return y.map({name: i for i, name in enumerate(C.CLASSES)}).astype(int)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split so every class keeps its share in both parts."""
    train, test = train_test_split(
        df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
