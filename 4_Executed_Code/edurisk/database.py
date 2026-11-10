"""SQLite storage that mirrors the ER diagram: course, student, semester_record,
app_user, risk_assessment, intervention.

The dashboard reads from and writes to this file, the same way the module would
plug into Edullent's own database.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from . import config as C

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE course (
    course_id      INTEGER PRIMARY KEY,
    course_name    TEXT NOT NULL
);

CREATE TABLE student (
    student_id          INTEGER PRIMARY KEY,
    course_id           INTEGER NOT NULL REFERENCES course(course_id),
    gender              TEXT,
    age_at_enrollment   INTEGER,
    admission_grade     REAL,
    scholarship_holder  INTEGER,
    displaced           INTEGER,
    debtor              INTEGER,
    tuition_up_to_date  INTEGER,
    international       INTEGER,
    cohort              TEXT NOT NULL CHECK (cohort IN ('history', 'current')),
    actual_outcome      TEXT CHECK (actual_outcome IN ('Dropout', 'Enrolled', 'Graduate'))
);

CREATE TABLE semester_record (
    record_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id          INTEGER NOT NULL REFERENCES student(student_id),
    semester_no         INTEGER NOT NULL CHECK (semester_no IN (1, 2)),
    units_credited      INTEGER,
    units_enrolled      INTEGER,
    units_evaluated     INTEGER,
    units_approved      INTEGER,
    units_without_eval  INTEGER,
    average_grade       REAL,
    UNIQUE (student_id, semester_no)
);

CREATE TABLE app_user (
    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name  TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('Principal', 'Teacher', 'Counsellor', 'Admin'))
);

CREATE TABLE risk_assessment (
    assessment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id     INTEGER NOT NULL REFERENCES student(student_id),
    checkpoint     TEXT NOT NULL CHECK (checkpoint IN ('enrollment', 'semester1', 'semester2')),
    model_name     TEXT NOT NULL,
    p_dropout      REAL NOT NULL,
    p_enrolled     REAL NOT NULL,
    p_graduate     REAL NOT NULL,
    risk_tier      TEXT NOT NULL CHECK (risk_tier IN ('High', 'Medium', 'Low')),
    reasons_json   TEXT,
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (student_id, checkpoint)
);

CREATE TABLE intervention (
    intervention_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id    INTEGER NOT NULL REFERENCES risk_assessment(assessment_id),
    assigned_to      INTEGER REFERENCES app_user(user_id),
    action           TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'Open' CHECK (status IN ('Open', 'In progress', 'Closed')),
    note             TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_tier ON risk_assessment (checkpoint, risk_tier);
"""

DEMO_USERS = [("Demo Principal", "Principal"), ("Demo Class Teacher", "Teacher"),
              ("Demo Counsellor", "Counsellor"), ("Demo Admin", "Admin")]


def connect(path: Path | str = C.DB_FILE) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create(path: Path | str = C.DB_FILE) -> None:
    """(Re)create an empty database with the schema and demo users."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with connect(path) as con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO app_user (full_name, role) VALUES (?, ?)", DEMO_USERS)


def load_students(df: pd.DataFrame, current_ids: set[int],
                  path: Path | str = C.DB_FILE) -> None:
    """Insert courses, students and their two semester records."""
    with connect(path) as con:
        con.executemany("INSERT INTO course VALUES (?, ?)",
                        sorted((int(k), v) for k, v in C.COURSE_NAMES.items()))
        students = [(
            int(r.student_id), int(r.course), "Male" if r.gender == 1 else "Female",
            int(r.age_at_enrollment), float(r.admission_grade), int(r.scholarship_holder),
            int(r.displaced), int(r.debtor), int(r.tuition_up_to_date), int(r.international),
            "current" if r.student_id in current_ids else "history", r.target,
        ) for r in df.itertuples()]
        con.executemany("INSERT INTO student VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", students)
        records = []
        for r in df.itertuples():
            for s in (1, 2):
                g = lambda f: getattr(r, f"sem{s}_{f}")
                records.append((int(r.student_id), s, int(g("credited")), int(g("enrolled")),
                                int(g("evaluations")), int(g("approved")),
                                int(g("without_eval")), float(g("grade"))))
        con.executemany(
            "INSERT INTO semester_record (student_id, semester_no, units_credited, "
            "units_enrolled, units_evaluated, units_approved, units_without_eval, "
            "average_grade) VALUES (?,?,?,?,?,?,?,?)", records)


def save_assessments(pred: pd.DataFrame, checkpoint: str, model_name: str,
                     path: Path | str = C.DB_FILE) -> None:
    """pred columns: student_id, p_dropout, p_enrolled, p_graduate, risk_tier, reasons."""
    rows = [(int(r.student_id), checkpoint, model_name, float(r.p_dropout),
             float(r.p_enrolled), float(r.p_graduate), r.risk_tier, json.dumps(r.reasons))
            for r in pred.itertuples()]
    with connect(path) as con:
        con.executemany(
            "INSERT INTO risk_assessment (student_id, checkpoint, model_name, p_dropout, "
            "p_enrolled, p_graduate, risk_tier, reasons_json) VALUES (?,?,?,?,?,?,?,?)", rows)


def log_intervention(assessment_id: int, user_id: int, action: str, note: str = "",
                     path: Path | str = C.DB_FILE) -> int:
    with connect(path) as con:
        cur = con.execute(
            "INSERT INTO intervention (assessment_id, assigned_to, action, note) "
            "VALUES (?,?,?,?)", (assessment_id, user_id, action, note))
        return int(cur.lastrowid)


def table_counts(path: Path | str = C.DB_FILE) -> dict:
    tables = ["course", "student", "semester_record", "app_user",
              "risk_assessment", "intervention"]
    with connect(path) as con:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
