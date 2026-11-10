"""EduRisk end-to-end pipeline.

    python run_pipeline.py            # full run (about 2-4 minutes on a laptop)
    python run_pipeline.py --quick    # 2-fold CV and no tuning, for a fast smoke test

Steps: load + validate data -> split -> EDA figures -> for each checkpoint
(enrollment, semester 1, semester 2) compare 5 models with 5-fold CV, tune XGBoost,
pick the best tree model, test it on the held-out 20% -> SHAP explanations, risk tiers,
fairness check -> write predictions to SQLite for the dashboard -> metrics.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from edurisk import config as C
from edurisk import data, database, evaluate, explain, fairness, features, plots
from edurisk import models as M


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def write_data_dictionary(df: pd.DataFrame) -> None:
    groups = {**{f: "Enrollment (demographic / socio-economic / admission)"
                 for f in C.ENROLLMENT_FEATURES},
              **{f: "Academic - Semester 1" for f in C.SEM1_RAW},
              **{f: "Academic - Semester 2" for f in C.SEM2_RAW}}
    rows = [{"original_column": raw, "clean_name": clean,
             "description": C.PRETTY.get(clean, "Final outcome: Dropout / Enrolled / Graduate"
                                         if clean == C.TARGET else clean),
             "group": groups.get(clean, "Target" if clean == C.TARGET else ""),
             "min": df[clean].min() if clean != C.TARGET else "",
             "max": df[clean].max() if clean != C.TARGET else ""}
            for raw, clean in C.RAW_TO_CLEAN.items()]
    pd.DataFrame(rows).to_csv(C.CD_DIR / "3_Datasets" / "data_dictionary.csv", index=False)


def main(quick: bool = False) -> dict:
    t0 = time.time()
    if quick:
        C.CV_FOLDS = 2
    for d in (C.OUTPUT_DIR, C.FIG_DIR, C.MODEL_DIR, C.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    F = C.FIG_DIR

    # 1. data ---------------------------------------------------------------
    log("Loading and validating data")
    df = data.load_raw()
    summary = data.validate(df)
    log(f"  {summary['rows']} students, classes {summary['class_counts']}")
    write_data_dictionary(df)
    train, test = data.split(df)
    train.to_csv(C.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(C.PROCESSED_DIR / "test.csv", index=False)
    clean_to_raw = {v: k for k, v in C.RAW_TO_CLEAN.items()}
    (test.head(25).drop(columns=["student_id", C.TARGET]).rename(columns=clean_to_raw)
         .to_csv(C.CD_DIR / "3_Datasets" / "sample_new_students.csv", sep=";", index=False))
    log(f"  train {len(train)} / test {len(test)} (stratified, seed {C.SEED})")

    # 2. EDA ----------------------------------------------------------------
    log("EDA figures")
    plots.class_distribution(df, F / "01_class_distribution.png")
    plots.eda_key_features(df, F / "02_key_features_by_outcome.png")
    plots.dropout_by_course(df, F / "03_dropout_by_course.png")

    # 3. models per checkpoint ---------------------------------------------
    cv, tuning, test_scores, chosen, test_proba = {}, {}, {}, {}, {}
    for cp in C.CHECKPOINTS:
        log(f"Checkpoint '{cp}' ({len(C.CHECKPOINTS[cp])} features)")
        X_tr, y_tr = features.make_xy(train, cp)
        X_te, y_te = features.make_xy(test, cp)

        best_xgb, tuning[cp] = {}, []
        if not quick:
            for params in M.XGB_GRID:
                res = evaluate.cross_validate("XGBoost", X_tr, y_tr, params)
                tuning[cp].append({**params, "cv_macro_f1": res["mean"]["macro_f1"]})
            best_xgb = {k: v for k, v in max(tuning[cp], key=lambda r: r["cv_macro_f1"]).items()
                        if k != "cv_macro_f1"}
            log(f"  tuned XGBoost: {best_xgb}")

        cv[cp], test_scores[cp] = {}, {}
        fitted = {}
        for name in M.MODEL_NAMES:
            params = best_xgb if name == "XGBoost" else None
            cv[cp][name] = evaluate.cross_validate(name, X_tr, y_tr, params)
            model = M.fit(M.build(name, list(X_tr.columns), params), name, X_tr, y_tr)
            fitted[name] = model
            test_scores[cp][name] = evaluate.metrics(y_te, model.predict_proba(X_te))
            log(f"  {name:<20} CV macro-F1 {cv[cp][name]['mean']['macro_f1']:.3f}  "
                f"test macro-F1 {test_scores[cp][name]['macro_f1']:.3f}")

        # deploy the best tree model (needs TreeSHAP for per-student reasons)
        name = max(M.TREE_MODELS, key=lambda m: cv[cp][m]["mean"]["macro_f1"])
        chosen[cp] = {"model": name, "xgb_params": best_xgb if name == "XGBoost" else None}
        joblib.dump({"model": fitted[name], "features": list(X_tr.columns), "name": name,
                     "checkpoint": cp}, C.MODEL_DIR / f"edurisk_{cp}.joblib")
        test_proba[cp] = (fitted[name], X_te, y_te, fitted[name].predict_proba(X_te))
        log(f"  deployed model: {name}")

    # 4. figures that compare checkpoints ----------------------------------
    plots.model_comparison(cv, F / "04_model_comparison_cv.png")
    curves = {}
    for cp, (_, _, y_te, proba) in test_proba.items():
        fpr, tpr, _ = roc_curve(y_te == C.DROPOUT, proba[:, C.DROPOUT])
        curves[cp] = (fpr, tpr, test_scores[cp][chosen[cp]["model"]]["dropout_auc"],
                      chosen[cp]["model"])
    plots.roc_dropout(curves, F / "05_roc_dropout.png")
    for cp in ("semester1", "semester2"):
        _, _, y_te, proba = test_proba[cp]
        plots.confusion(evaluate.confusion(y_te, proba),
                        f"Confusion matrix\n{C.CHECKPOINT_LABELS[cp]} - "
                        f"{chosen[cp]['model']}, test set",
                        F / f"06_confusion_{cp}.png")

    # 5. database + per-student explanations for every checkpoint ----------
    log("Building SQLite database and per-student explanations")
    database.create()
    database.load_students(df, set(test["student_id"]))
    explanations = {}
    for cp, (model, X_te, y_te, proba) in test_proba.items():
        shap_drop, base = explain.dropout_shap(model, X_te)
        explanations[cp] = (shap_drop, base)
        pred = pd.DataFrame({
            "student_id": test["student_id"], "p_dropout": proba[:, 0],
            "p_enrolled": proba[:, 1], "p_graduate": proba[:, 2],
            "risk_tier": evaluate.risk_tier(proba[:, 0]),
            "actual_outcome": test[C.TARGET],
        })
        pred["reasons"] = [explain.top_reasons(shap_drop[i], X_te.iloc[i])
                           for i in range(len(X_te))]
        database.save_assessments(pred, cp, chosen[cp]["model"])
        pred.assign(reasons=pred["reasons"].map(lambda r: " | ".join(x["text"] for x in r))) \
            .to_csv(C.PROCESSED_DIR / f"predictions_{cp}.csv", index=False)

    # 6. deep dive on the primary (early-warning) checkpoint ---------------
    cp = C.PRIMARY_CHECKPOINT
    model, X_te, y_te, proba = test_proba[cp]
    shap_drop, base = explanations[cp]
    importance = explain.global_importance(shap_drop, list(X_te.columns))
    plots.shap_bar(importance, F / "07_shap_importance.png")
    plots.shap_beeswarm(shap_drop, X_te, F / "08_shap_beeswarm.png")

    p = proba[:, C.DROPOUT]
    candidates = np.where((p >= C.HIGH_RISK) & (y_te.to_numpy() == C.DROPOUT))[0]
    pick = int(candidates[np.argmin(np.abs(p[candidates] - 0.80))])
    sid = int(test.loc[pick, "student_id"])
    plots.shap_waterfall(shap_drop[pick], base, X_te.iloc[pick],
                         F / "09_shap_one_student.png", f"student #{sid}")

    tiers = evaluate.tier_table(y_te, p)
    plots.risk_tiers(tiers, F / "10_risk_tiers.png")
    fair = fairness.group_report(test, y_te, p)
    plots.fairness(fair, F / "11_fairness.png")

    # 7. save everything ---------------------------------------------------
    import shap, sklearn, xgboost  # noqa: E401  (versions for the report)
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "quick_mode": quick,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "shap": shap.__version__,
                        "platform": platform.platform()},
        "data": summary, "split": {"train": len(train), "test": len(test)},
        "cv": cv, "xgb_tuning": tuning, "test": test_scores, "deployed": chosen,
        "primary_checkpoint": cp,
        "example_student": {"student_id": sid, "p_dropout": float(p[pick]),
                            "reasons": explain.top_reasons(shap_drop[pick], X_te.iloc[pick], 5)},
        "shap_top": importance.head(15).to_dict(orient="records"),
        "risk_tiers": tiers.to_dict(orient="records"),
        "fairness": fair.to_dict(orient="records"),
        "fairness_recall_gap": fairness.recall_gap(fair).to_dict(orient="records"),
        "db_counts": database.table_counts(),
    }
    C.METRICS_FILE.write_text(json.dumps(result, indent=2, default=float))
    log(f"Done in {result['runtime_seconds']} s -> {C.METRICS_FILE.relative_to(C.CODE_DIR)}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="2-fold CV, no tuning")
    main(quick=ap.parse_args().quick)
