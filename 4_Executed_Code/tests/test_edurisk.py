"""Automated tests for EduRisk.   Run:  python -m pytest -q tests

Unit tests use only the dataset; the integration tests at the bottom need the outputs
of `python run_pipeline.py` and are skipped if those are missing.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edurisk import config as C                                      # noqa: E402
from edurisk import data, evaluate, explain, features, scoring       # noqa: E402
from edurisk import models as M                                      # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data.load_raw()


# ---------------------------------------------------------------- data
def test_dataset_shape_and_quality(df):
    summary = data.validate(df)
    assert summary["rows"] == 4424
    assert df.shape[1] == len(C.RAW_TO_CLEAN) + 1            # + student_id
    assert summary["class_counts"] == {"Dropout": 1421, "Enrolled": 794, "Graduate": 2209}
    assert not df.isna().any().any()


def test_split_is_stratified_and_disjoint(df):
    train, test = data.split(df)
    assert len(train) + len(test) == len(df)
    assert set(train["student_id"]).isdisjoint(test["student_id"])
    share = lambda d: (d[C.TARGET] == "Dropout").mean()
    assert abs(share(train) - share(test)) < 0.01


def test_split_is_reproducible(df):
    a, _ = data.split(df)
    b, _ = data.split(df)
    assert a["student_id"].tolist() == b["student_id"].tolist()


# ---------------------------------------------------------------- features
def test_checkpoints_never_use_future_data():
    enrol, s1, s2 = (set(C.CHECKPOINTS[k]) for k in ("enrollment", "semester1", "semester2"))
    assert enrol < s1 < s2
    assert not any(f.startswith(("sem1", "sem2")) for f in enrol)
    assert not any(f.startswith("sem2") or f == "grade_trend" for f in s1)


def test_engineered_rates_are_valid(df):
    X, y = features.make_xy(df, "semester2")
    for col in ("sem1_approval_rate", "sem2_approval_rate"):
        assert X[col].between(0, 1).all()
    assert set(y.unique()) == {0, 1, 2}


def test_zero_enrolled_units_give_zero_rate():
    row = pd.DataFrame({"sem1_approved": [0], "sem1_enrolled": [0], "sem2_approved": [3],
                        "sem2_enrolled": [0], "sem1_grade": [0.0], "sem2_grade": [0.0]})
    out = features.add_engineered(row)
    assert out.loc[0, "sem1_approval_rate"] == 0 and out.loc[0, "sem2_approval_rate"] == 0


# ---------------------------------------------------------------- models / evaluation
@pytest.mark.parametrize("name", M.MODEL_NAMES)
def test_every_model_trains_and_outputs_probabilities(df, name):
    train, test = data.split(df)
    X_tr, y_tr = features.make_xy(train, "semester1")
    X_te, _ = features.make_xy(test.head(50), "semester1")
    model = M.fit(M.build(name, list(X_tr.columns)), name, X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_risk_tier_boundaries():
    tiers = evaluate.risk_tier([0.0, 0.249, 0.25, 0.499, 0.5, 1.0])
    assert tiers.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_metrics_perfect_and_baseline():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    m = evaluate.metrics(y, perfect)
    assert m["accuracy"] == 1 and m["dropout_recall"] == 1 and m["dropout_auc"] == 1


def test_reasons_only_list_risk_increasing_features():
    x = pd.Series({"sem1_approval_rate": 0.0, "age_at_enrollment": 30, "gender": 1})
    reasons = explain.top_reasons(np.array([0.9, -0.2, 0.1]), x, k=3)
    assert [r["feature"] for r in reasons] == ["sem1_approval_rate", "gender"]
    assert reasons[0]["text"] == "Sem 1 pass rate = 0%"


def test_upload_validation_handles_bad_files(df):
    good = df.head(4).drop(columns=["student_id", C.TARGET])
    to_bytes = lambda d: d.to_csv(index=False).encode()
    with pytest.raises(ValueError, match="empty"):
        scoring.read_any(b"")
    with pytest.raises(ValueError, match="no student rows"):
        scoring.prepare(scoring.read_any(to_bytes(good.head(0))), "semester1")
    broken = good.astype(object).copy()
    broken.loc[broken.index[0], "sem1_grade"] = "oops"        # non-numeric value
    broken.loc[broken.index[1], "course"] = None               # blank value
    rows, skipped = scoring.prepare(scoring.read_any(to_bytes(broken)), "semester1")
    assert skipped == [1, 2] and len(rows) == 2
    all_bad = broken.copy()
    all_bad["course"] = None
    with pytest.raises(ValueError, match="No row could be scored"):
        scoring.prepare(scoring.read_any(to_bytes(all_bad)), "semester1")


# ---------------------------------------------------------------- integration (needs pipeline outputs)
needs_outputs = pytest.mark.skipif(not C.DB_FILE.exists() or not C.METRICS_FILE.exists(),
                                   reason="run `python run_pipeline.py` first")


@needs_outputs
def test_database_matches_er_design():
    with sqlite3.connect(C.DB_FILE) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"course", "student", "semester_record", "app_user", "risk_assessment",
                "intervention"} <= tables
        assert con.execute("SELECT COUNT(*) FROM student").fetchone()[0] == 4424
        assert con.execute("SELECT COUNT(*) FROM semester_record").fetchone()[0] == 2 * 4424
        orphans = con.execute("""SELECT COUNT(*) FROM risk_assessment r
                                 LEFT JOIN student s ON s.student_id = r.student_id
                                 WHERE s.student_id IS NULL""").fetchone()[0]
        assert orphans == 0
        per_cp = con.execute("SELECT checkpoint, COUNT(*) FROM risk_assessment "
                             "GROUP BY checkpoint").fetchall()
        assert dict(per_cp) == {"enrollment": 885, "semester1": 885, "semester2": 885}


@needs_outputs
def test_deployed_models_beat_the_baseline():
    m = json.loads(C.METRICS_FILE.read_text())
    for cp in C.CHECKPOINTS:
        chosen = m["deployed"][cp]["model"]
        assert m["test"][cp][chosen]["macro_f1"] > m["test"][cp]["Majority baseline"]["macro_f1"] + 0.2
    # more data later in the year should not make the warning worse
    auc = [m["test"][cp][m["deployed"][cp]["model"]]["dropout_auc"] for cp in C.CHECKPOINTS]
    assert auc[0] < auc[1] <= auc[2] + 0.01


@needs_outputs
def test_scoring_new_upload_in_original_layout():
    sample = (C.CD_DIR / "3_Datasets" / "sample_new_students.csv").read_bytes()
    new = scoring.read_any(sample)
    assert scoring.missing_columns(new, "semester1") == []
    out = scoring.score(new, "semester1")
    assert len(out) == 25
    assert out["p_dropout"].between(0, 1).all()
    assert out["p_dropout"].is_monotonic_decreasing
