"""Candidate models and how each one is trained.

All models are compared on exactly the same folds. Class imbalance (only ~18% of
students are 'Enrolled') is handled with balanced class weights rather than by
inventing synthetic students.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from .features import nominal_in

MODEL_NAMES = ["Majority baseline", "Logistic Regression", "Decision Tree",
               "Random Forest", "XGBoost"]
TREE_MODELS = ["Random Forest", "XGBoost"]          # these can be explained with TreeSHAP

XGB_DEFAULTS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1.0)


def build(name: str, columns: list[str], xgb_params: dict | None = None):
    """Return an untrained model. `columns` is needed to one-hot the nominal codes."""
    if name == "Majority baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "Logistic Regression":
        nominal = nominal_in(columns)
        numeric = [c for c in columns if c not in nominal]
        pre = ColumnTransformer([
            ("nominal", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
             nominal),
            ("numeric", StandardScaler(), numeric),
        ])
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=4000, C=0.5,
                                                    class_weight="balanced"))])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                      class_weight="balanced", random_state=C.SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=C.SEED)
    if name == "XGBoost":
        params = {**XGB_DEFAULTS, **(xgb_params or {})}
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                             tree_method="hist", random_state=C.SEED, n_jobs=4, **params)
    raise KeyError(f"Unknown model '{name}'")


def fit(model, name: str, X, y):
    """Train. XGBoost has no class_weight option, so it gets balanced sample weights."""
    if name == "XGBoost":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


# Small, honest tuning grid for XGBoost (6 combinations, judged by 5-fold CV macro-F1).
XGB_GRID = [
    {"max_depth": d, "learning_rate": lr}
    for d in (3, 4, 6) for lr in (0.05, 0.10)
]
