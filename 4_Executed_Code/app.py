"""EduRisk dashboard (Streamlit).

    streamlit run app.py

Run `python run_pipeline.py` once first - it creates the models and the SQLite database
this dashboard reads. The 885 held-out test students play the role of the current cohort.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from edurisk import config as C
from edurisk import database, scoring

st.set_page_config(page_title="EduRisk - Student Early Warning", page_icon="🎓", layout="wide")

TIER_BADGE = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
ACTIONS = ["Counselling session", "Parent / guardian meeting", "Extra tutoring",
           "Fee / scholarship guidance", "Mentor assigned", "Attendance follow-up"]


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(C.DB_FILE) as con:
        return pd.read_sql_query(sql, con, params=params)


@st.cache_data
def metrics() -> dict:
    return json.loads(C.METRICS_FILE.read_text())


def reasons_text(js: str) -> str:
    return " · ".join(r["text"] for r in json.loads(js)) if js else ""


if not C.DB_FILE.exists() or not C.METRICS_FILE.exists():
    st.error("Run `python run_pipeline.py` first - the database and models are missing.")
    st.stop()

M = metrics()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎓 EduRisk")
st.sidebar.caption("Early-warning module for the Edullent school platform")
users = query("SELECT user_id, full_name, role FROM app_user")
user_label = st.sidebar.selectbox("Signed in as", users["full_name"] + " (" + users["role"] + ")")
user_id = int(users.loc[(users["full_name"] + " (" + users["role"] + ")") == user_label,
                        "user_id"].iloc[0])
checkpoint = st.sidebar.radio("Data available", list(C.CHECKPOINTS),
                              index=list(C.CHECKPOINTS).index(C.PRIMARY_CHECKPOINT),
                              format_func=C.CHECKPOINT_LABELS.get)
tiers = st.sidebar.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"],
                               format_func=TIER_BADGE.get)
courses = query("SELECT course_id, course_name FROM course ORDER BY course_name")
course_pick = st.sidebar.multiselect("Course", courses["course_name"].tolist())

base = query("""
    SELECT r.assessment_id, s.student_id, c.course_name, s.gender, s.age_at_enrollment,
           s.scholarship_holder, s.tuition_up_to_date, r.p_dropout, r.p_enrolled,
           r.p_graduate, r.risk_tier, r.reasons_json, r.model_name, s.actual_outcome
    FROM risk_assessment r
    JOIN student s ON s.student_id = r.student_id
    JOIN course  c ON c.course_id  = s.course_id
    WHERE r.checkpoint = ?""", (checkpoint,))
view = base[base["risk_tier"].isin(tiers)] if tiers else base
if course_pick:
    view = view[view["course_name"].isin(course_pick)]
view = view.sort_values("p_dropout", ascending=False)

tab_overview, tab_list, tab_student, tab_upload, tab_model = st.tabs(
    ["Overview", "At-risk list", "Student profile", "Score new students", "Model insight"])

# ---------------------------------------------------------------- overview
with tab_overview:
    st.subheader(f"Current cohort - {C.CHECKPOINT_LABELS[checkpoint]}")
    counts = base["risk_tier"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students scored", f"{len(base):,}")
    k2.metric("🔴 High risk", int(counts.get("High", 0)))
    k3.metric("🟠 Medium risk", int(counts.get("Medium", 0)))
    k4.metric("🟢 Low risk", int(counts.get("Low", 0)))
    model_name = M["deployed"][checkpoint]["model"]
    t = M["test"][checkpoint][model_name]
    st.caption(f"Model: **{model_name}** · test accuracy {t['accuracy']:.1%} · "
               f"dropouts caught {t['dropout_recall']:.1%} · Dropout AUC {t['dropout_auc']:.3f}")
    by_course = (base.assign(high=base["risk_tier"].eq("High"))
                     .groupby("course_name")["high"].sum().sort_values(ascending=False))
    st.markdown("**High-risk students per course**")
    st.bar_chart(by_course, horizontal=True, color="#d03b3b")

# ---------------------------------------------------------------- list
with tab_list:
    st.subheader(f"{len(view)} students match the filters")
    table = view.assign(
        tier=view["risk_tier"].map(TIER_BADGE),
        dropout_risk=(view["p_dropout"] * 100).round(1),
        why=view["reasons_json"].map(reasons_text),
    )[["student_id", "tier", "dropout_risk", "course_name", "age_at_enrollment", "why"]]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"dropout_risk": st.column_config.ProgressColumn(
                     "Dropout risk %", min_value=0, max_value=100, format="%.1f")})
    st.download_button("Download list (CSV)", table.to_csv(index=False),
                       file_name=f"edurisk_{checkpoint}_list.csv")

# ---------------------------------------------------------------- student
with tab_student:
    if view.empty:
        st.info("No student matches the filters.")
    else:
        sid = st.selectbox("Student", view["student_id"],
                           format_func=lambda s: f"#{s} - {TIER_BADGE[view.set_index('student_id').loc[s, 'risk_tier']]}")
        row = base.set_index("student_id").loc[sid]
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"### Student #{sid} · {TIER_BADGE[row['risk_tier']]}")
            st.write(f"{row['course_name']} · {row['gender']} · age {row['age_at_enrollment']}")
            st.bar_chart(pd.Series({"Dropout": row["p_dropout"], "Enrolled": row["p_enrolled"],
                                    "Graduate": row["p_graduate"]}, name="probability"),
                         color="#2a78d6")
            st.markdown("**Why the model flagged this student**")
            for r in json.loads(row["reasons_json"]):
                st.write(f"- {r['text']}  (+{r['impact']:.2f})")
        with right:
            st.markdown("**Semester records**")
            st.dataframe(query("SELECT semester_no, units_enrolled, units_evaluated, "
                               "units_approved, average_grade FROM semester_record "
                               "WHERE student_id = ? ORDER BY semester_no", (int(sid),)),
                         hide_index=True)
            with st.form("intervention"):
                st.markdown("**Log an intervention**")
                action = st.selectbox("Action", ACTIONS)
                note = st.text_area("Note", max_chars=300)
                if st.form_submit_button("Save"):
                    database.log_intervention(int(row["assessment_id"]), user_id, action, note)
                    st.success("Saved.")
            st.dataframe(query("""SELECT i.created_at, u.full_name AS by_user, i.action,
                                         i.status, i.note
                                  FROM intervention i
                                  JOIN risk_assessment r ON r.assessment_id = i.assessment_id
                                  LEFT JOIN app_user u ON u.user_id = i.assigned_to
                                  WHERE r.student_id = ? ORDER BY i.created_at DESC""",
                               (int(sid),)), hide_index=True)

# ---------------------------------------------------------------- upload
with tab_upload:
    st.subheader("Score a new batch of students")
    st.caption("Upload a CSV in the original dataset layout (a sample is in "
               "3_Datasets/sample_new_students.csv). The Target column is not needed.")
    sample = C.CD_DIR / "3_Datasets" / "sample_new_students.csv"
    if sample.exists():
        st.download_button("Download a sample file to try", sample.read_bytes(),
                           file_name="sample_new_students.csv", mime="text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            new, skipped = scoring.prepare(scoring.read_any(up.getvalue()), checkpoint)
        except ValueError as exc:            # empty, unreadable, missing columns, no valid row
            st.error(str(exc))
        else:
            if skipped:
                more = " ..." if len(skipped) > 10 else ""
                st.warning(f"{len(skipped)} row(s) skipped: a required value is blank or not a "
                           f"number (student ids {skipped[:10]}{more}).")
            scored = scoring.score(new, checkpoint)
            scored["risk_tier"] = scored["risk_tier"].map(TIER_BADGE)
            st.dataframe(scored, hide_index=True, use_container_width=True)
            st.download_button("Download scores (CSV)", scored.to_csv(index=False),
                               file_name="edurisk_scores.csv")

# ---------------------------------------------------------------- model insight
with tab_model:
    st.subheader("How reliable is the warning?")
    rows = [{"Checkpoint": C.CHECKPOINT_LABELS[cp], "Model": M["deployed"][cp]["model"],
             **{k: round(v, 3) for k, v in M["test"][cp][M["deployed"][cp]["model"]].items()}}
            for cp in C.CHECKPOINTS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(str(C.FIG_DIR / "07_shap_importance.png"))
    c2.image(str(C.FIG_DIR / "10_risk_tiers.png"))
    st.image(str(C.FIG_DIR / "11_fairness.png"))
    st.caption("Fairness check on the test set: the warning catches fewer dropouts among "
               "scholarship holders and students aged 17-20, so counsellors should not treat "
               "'Low' as 'safe' for these groups.")
