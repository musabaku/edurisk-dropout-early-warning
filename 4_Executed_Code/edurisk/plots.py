"""All figures used in the report and the slides. Every function saves one PNG.

Colour rules: one fixed colour per outcome class; ordered checkpoints use one blue hue
light->dark; risk tiers use status colours and always carry a text label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CLASS_COLOR = {"Dropout": "#eb6834", "Enrolled": "#2a78d6", "Graduate": "#1baf7a"}
CHECKPOINT_COLOR = {"enrollment": "#86b6ef", "semester1": "#2a78d6", "semester2": "#104281"}
TIER_COLOR = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c"}
SERIES = ["#2a78d6", "#eb6834"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def class_distribution(df: pd.DataFrame, path: Path) -> Path:
    counts = df[C.TARGET].value_counts().reindex(C.CLASSES)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(C.CLASSES[::-1], counts[::-1], color=[CLASS_COLOR[c] for c in C.CLASSES[::-1]],
            height=0.6)
    for i, c in enumerate(C.CLASSES[::-1]):
        ax.text(counts[c] + 25, i, f"{counts[c]:,}  ({counts[c] / counts.sum():.1%})",
                va="center", color=INK)
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_xlabel("Students")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Final outcome of {counts.sum():,} students", loc="left")
    return _save(fig, path)


def eda_key_features(df: pd.DataFrame, path: Path) -> Path:
    d = df.copy()
    d["sem1_pass_rate"] = np.where(d["sem1_enrolled"] > 0,
                                   d["sem1_approved"] / d["sem1_enrolled"].replace(0, 1), 0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [("sem1_pass_rate", "Semester 1 pass rate"),
              ("sem1_grade", "Semester 1 average grade (0-20)"),
              ("age_at_enrollment", "Age at enrollment")]
    for ax, (col, title) in zip(axes.flat, panels):
        data = [d.loc[d[C.TARGET] == c, col] for c in C.CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5})
        for patch, c in zip(bp["boxes"], C.CLASSES):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.85)
        ax.set_xticks([1, 2, 3], C.CLASSES)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    ax = axes.flat[3]
    rate = d.groupby(C.TARGET)["tuition_up_to_date"].mean().reindex(C.CLASSES) * 100
    ax.bar(C.CLASSES, rate, color=[CLASS_COLOR[c] for c in C.CLASSES], width=0.55)
    for i, v in enumerate(rate):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", color=INK)
    ax.set_ylim(0, 110)
    ax.set_title("Tuition fees up to date (%)", loc="left")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def dropout_by_course(df: pd.DataFrame, path: Path) -> Path:
    g = (df.assign(drop=df[C.TARGET] == "Dropout")
           .groupby("course").agg(rate=("drop", "mean"), n=("drop", "size")).reset_index())
    g["name"] = g["course"].map(C.COURSE_NAMES)
    g = g.sort_values("rate")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(g["name"], g["rate"] * 100, color=CLASS_COLOR["Dropout"], height=0.65)
    for i, (r, n) in enumerate(zip(g["rate"], g["n"])):
        ax.text(r * 100 + 1, i, f"{r:.0%}  (n={n})", va="center", color=INK2, fontsize=8.5)
    overall = (df[C.TARGET] == "Dropout").mean() * 100
    ax.axvline(overall, color=MUTED, linestyle="--", linewidth=1)
    ax.text(overall + 0.5, -0.9, f"overall {overall:.0f}%", color=MUTED, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Dropout rate (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Dropout rate by course", loc="left")
    return _save(fig, path)


def model_comparison(cv: dict, path: Path) -> Path:
    """cv[checkpoint][model] = {'mean': {...}, 'std': {...}}"""
    from .models import MODEL_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(MODEL_NAMES))
    w = 0.26
    for ax, metric, title in [(axes[0], "macro_f1", "Macro-F1 (all three classes)"),
                              (axes[1], "dropout_recall", "Dropout recall (share of dropouts caught)")]:
        for i, cp in enumerate(C.CHECKPOINTS):
            means = [cv[cp][m]["mean"][metric] for m in MODEL_NAMES]
            stds = [cv[cp][m]["std"][metric] for m in MODEL_NAMES]
            ax.bar(x + (i - 1) * w, means, w - 0.03, yerr=stds, capsize=2,
                   color=CHECKPOINT_COLOR[cp], label=C.CHECKPOINT_LABELS[cp],
                   error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.set_xticks(x, [m.replace(" ", "\n") for m in MODEL_NAMES], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (5-fold CV mean, +/- 1 std)")
    axes[0].legend(loc="upper left", fontsize=9)
    return _save(fig, path)


def confusion(cm: np.ndarray, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    color="white" if row_pct[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(3), C.CLASSES)
    ax.set_yticks(range(3), C.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, path)


def roc_dropout(curves: dict, path: Path) -> Path:
    """curves[checkpoint] = (fpr, tpr, auc, model_name)"""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for cp, (fpr, tpr, auc, name) in curves.items():
        ax.plot(fpr, tpr, color=CHECKPOINT_COLOR[cp], linewidth=2,
                label=f"{C.CHECKPOINT_LABELS[cp]} ({name}) AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (dropouts caught)")
    ax.set_title("Dropout vs rest - ROC on the test set", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    return _save(fig, path)


def shap_bar(importance: pd.DataFrame, path: Path, top: int = 12) -> Path:
    d = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["label"], d["mean_abs_shap"], color=SERIES[0], height=0.65)
    ax.set_xlabel("Mean |SHAP value| for the Dropout score")
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives dropout risk (after Semester 1)", loc="left")
    return _save(fig, path)


def shap_beeswarm(shap_drop: np.ndarray, X: pd.DataFrame, path: Path, top: int = 12) -> Path:
    import shap
    Xp = X.rename(columns=lambda c: C.PRETTY.get(c, c))
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_drop, Xp, max_display=top, show=False, plot_size=None)
    fig = plt.gcf()
    fig.axes[0].set_title("Per-student SHAP values (red = high feature value)", loc="left",
                          fontsize=11)
    return _save(fig, path)


def shap_waterfall(shap_row: np.ndarray, base: float, x_row: pd.Series, path: Path,
                   student_label: str) -> Path:
    import shap
    exp = shap.Explanation(values=shap_row, base_values=base, data=x_row.to_numpy(),
                           feature_names=[C.PRETTY.get(c, c) for c in x_row.index])
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Why {student_label} was flagged - SHAP contributions to the Dropout class "
                 "raw score (margin before softmax)", x=0.02, ha="left", fontsize=10.5,
                 fontweight="bold")
    return _save(fig, path)


def risk_tiers(tiers: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    names = tiers["tier"].tolist()
    colors = [TIER_COLOR[t] for t in names]
    axes[0].bar(names, tiers["students"], color=colors, width=0.55)
    for i, (n, s) in enumerate(zip(tiers["students"], tiers["share_of_students"])):
        axes[0].text(i, n + 5, f"{n} ({s:.0%})", ha="center", color=INK)
    axes[0].set_title("Students per risk tier", loc="left")
    axes[0].set_ylim(0, tiers["students"].max() * 1.2)
    axes[1].bar(names, tiers["dropout_rate_in_tier"] * 100, color=colors, width=0.55)
    for i, r in enumerate(tiers["dropout_rate_in_tier"]):
        axes[1].text(i, r * 100 + 2, f"{r:.0%}", ha="center", color=INK)
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Students in the tier who really dropped out", loc="left")
    for ax in axes:
        ax.set_xlabel("Risk tier")
        ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fairness(report: pd.DataFrame, path: Path) -> Path:
    d = report.copy()
    d["label"] = d["attribute"] + ": " + d["group"].astype(str)
    d = d.iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(y + 0.18, d["recall"], 0.34, color=SERIES[0], label="Recall (dropouts flagged High)")
    ax.barh(y - 0.18, d["false_positive_rate"], 0.34, color=SERIES[1],
            label="False positive rate (non-dropouts flagged High)")
    ax.set_yticks(y, d["label"])
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=2, fontsize=8.5)
    ax.set_title("Does the warning work equally for every group?", loc="left")
    return _save(fig, path)
