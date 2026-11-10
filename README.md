# EduRisk - explainable early warning of student dropout risk

Streamlit dashboard for an early-warning module designed for the Edullent school platform.
It predicts each student's outcome (Dropout / Enrolled / Graduate) at three checkpoints
(enrollment, after Semester 1, after Semester 2), explains every prediction with SHAP,
groups students into High / Medium / Low risk tiers, and logs interventions.

- App entry point: `4_Executed_Code/app.py`
- Retrain everything: `cd 4_Executed_Code && python run_pipeline.py`
- Tests: `python -m pytest -q tests`

Data: *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository,
doi:10.24432/C5MC89 (CC BY 4.0). The 885 held-out test students act as the current cohort.
