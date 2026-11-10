"""Central configuration: file locations, random seed, label names and feature groups.

Every other module imports its constants from here, so a path or a feature list is
changed in exactly one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent.parent          # .../4_Executed_Code
CD_DIR = CODE_DIR.parent                                     # .../CD
DATA_FILE = CD_DIR / "3_Datasets" / "student_dropout_uci697.csv"
PROCESSED_DIR = CD_DIR / "3_Datasets" / "processed"
OUTPUT_DIR = CODE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
DB_FILE = OUTPUT_DIR / "edurisk.sqlite"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# ---------------------------------------------------------------- experiment settings
SEED = 42
TEST_SIZE = 0.20          # the dataset authors recommend an 80/20 split
CV_FOLDS = 5

TARGET = "target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]   # index 0 = Dropout (the class we care about)
DROPOUT = 0

# Risk tiers on P(Dropout). Chosen so a counsellor sees a short "High" list first.
HIGH_RISK = 0.50
MEDIUM_RISK = 0.25

# ---------------------------------------------------------------- raw column -> clean name
RAW_TO_CLEAN = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_attendance",
    "Previous qualification": "prev_qualification",
    "Previous qualification (grade)": "prev_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation",
    "Father's occupation": "father_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "sem1_credited",
    "Curricular units 1st sem (enrolled)": "sem1_enrolled",
    "Curricular units 1st sem (evaluations)": "sem1_evaluations",
    "Curricular units 1st sem (approved)": "sem1_approved",
    "Curricular units 1st sem (grade)": "sem1_grade",
    "Curricular units 1st sem (without evaluations)": "sem1_without_eval",
    "Curricular units 2nd sem (credited)": "sem2_credited",
    "Curricular units 2nd sem (enrolled)": "sem2_enrolled",
    "Curricular units 2nd sem (evaluations)": "sem2_evaluations",
    "Curricular units 2nd sem (approved)": "sem2_approved",
    "Curricular units 2nd sem (grade)": "sem2_grade",
    "Curricular units 2nd sem (without evaluations)": "sem2_without_eval",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": TARGET,
}

# ---------------------------------------------------------------- feature groups
# Nominal codes (the number is a label, not a quantity) -> one-hot for linear models.
NOMINAL = [
    "marital_status", "application_mode", "course", "prev_qualification", "nationality",
    "mother_qualification", "father_qualification", "mother_occupation", "father_occupation",
]

ENROLLMENT_FEATURES = [
    "marital_status", "application_mode", "application_order", "course", "daytime_attendance",
    "prev_qualification", "prev_qualification_grade", "nationality", "mother_qualification",
    "father_qualification", "mother_occupation", "father_occupation", "admission_grade",
    "displaced", "special_needs", "debtor", "tuition_up_to_date", "gender",
    "scholarship_holder", "age_at_enrollment", "international",
    "unemployment_rate", "inflation_rate", "gdp",
]
SEM1_RAW = ["sem1_credited", "sem1_enrolled", "sem1_evaluations", "sem1_approved",
            "sem1_grade", "sem1_without_eval"]
SEM2_RAW = ["sem2_credited", "sem2_enrolled", "sem2_evaluations", "sem2_approved",
            "sem2_grade", "sem2_without_eval"]
SEM1_ENGINEERED = ["sem1_approval_rate"]
SEM2_ENGINEERED = ["sem2_approval_rate", "grade_trend"]

# Checkpoints: what the school knows at each moment of the first year.
CHECKPOINTS = {
    "enrollment": ENROLLMENT_FEATURES,
    "semester1": ENROLLMENT_FEATURES + SEM1_RAW + SEM1_ENGINEERED,
    "semester2": ENROLLMENT_FEATURES + SEM1_RAW + SEM1_ENGINEERED + SEM2_RAW + SEM2_ENGINEERED,
}
CHECKPOINT_LABELS = {
    "enrollment": "At enrollment",
    "semester1": "After Semester 1",
    "semester2": "After Semester 2",
}
PRIMARY_CHECKPOINT = "semester1"   # earliest point with marks -> the real early warning

# Human-readable names used in charts and in the dashboard.
PRETTY = {
    "marital_status": "Marital status", "application_mode": "Application mode",
    "application_order": "Application order (choice rank)", "course": "Course",
    "daytime_attendance": "Daytime attendance", "prev_qualification": "Previous qualification",
    "prev_qualification_grade": "Previous qualification grade", "nationality": "Nationality",
    "mother_qualification": "Mother's qualification", "father_qualification": "Father's qualification",
    "mother_occupation": "Mother's occupation", "father_occupation": "Father's occupation",
    "admission_grade": "Admission grade", "displaced": "Displaced (lives away from home)",
    "special_needs": "Special educational needs", "debtor": "Debtor",
    "tuition_up_to_date": "Tuition fees up to date", "gender": "Gender (1 = male)",
    "scholarship_holder": "Scholarship holder", "age_at_enrollment": "Age at enrollment",
    "international": "International student", "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate", "gdp": "GDP growth",
    "sem1_credited": "Sem 1 units credited", "sem1_enrolled": "Sem 1 units enrolled",
    "sem1_evaluations": "Sem 1 evaluations", "sem1_approved": "Sem 1 units passed",
    "sem1_grade": "Sem 1 average grade", "sem1_without_eval": "Sem 1 units not evaluated",
    "sem2_credited": "Sem 2 units credited", "sem2_enrolled": "Sem 2 units enrolled",
    "sem2_evaluations": "Sem 2 evaluations", "sem2_approved": "Sem 2 units passed",
    "sem2_grade": "Sem 2 average grade", "sem2_without_eval": "Sem 2 units not evaluated",
    "sem1_approval_rate": "Sem 1 pass rate", "sem2_approval_rate": "Sem 2 pass rate",
    "grade_trend": "Grade trend (Sem 2 - Sem 1)",
}

COURSE_NAMES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)",
}
