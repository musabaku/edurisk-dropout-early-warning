"""EduRisk - an explainable early-warning module for student dropout risk.

Modules
-------
config    paths, seed, label names, feature groups
data      load + clean the raw CSV, split train/test
features  build the feature table for one checkpoint (enrollment / semester1 / semester2)
models    the candidate models and how they are trained
evaluate  cross-validation, test metrics, risk tiers
explain   SHAP explanations (global + per student)
fairness  error rates per student group
database  SQLite schema and loading of predictions
plots     all figures
"""
__version__ = "1.0.0"
