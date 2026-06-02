# -*- coding: utf-8 -*-
"""
Breast Cancer Classification using Logistic Regression
Improved core ML module with cross-validation, ROC/AUC, hyperparameter tuning.
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kivy compatibility
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def load_data() -> tuple:
    """Load Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 1 = benign, 0 = malignant
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, data.feature_names.tolist()


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 10000,
    C: float = 1.0,
    solver: str = "lbfgs",
    cv_folds: int = 5,
) -> dict:
    """
    Full pipeline: load → split → scale → train → evaluate.
    Returns a results dict consumed by the GUI.
    """
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scaler + model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=max_iter, random_state=random_state, C=C, solver=solver
        )),
    ])
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Core metrics
    acc   = accuracy_score(y_test, y_pred) * 100
    prec  = precision_score(y_test, y_pred) * 100
    rec   = recall_score(y_test, y_pred) * 100
    f1    = f1_score(y_test, y_pred) * 100
    cm    = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"])

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr) * 100

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy") * 100

    # Feature importance (coefficients from the model step)
    model_step = pipeline.named_steps["model"]
    coefs = pd.Series(np.abs(model_step.coef_[0]), index=feature_names)
    top_features = coefs.nlargest(10).sort_values()

    return {
        "accuracy":     acc,
        "precision":    prec,
        "recall":       rec,
        "f1":           f1,
        "roc_auc":      roc_auc,
        "cv_mean":      cv_scores.mean(),
        "cv_std":       cv_scores.std(),
        "cv_scores":    cv_scores.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
        "fpr":          fpr,
        "tpr":          tpr,
        "top_features": top_features,
        "feature_names": feature_names,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "params": {
            "test_size": test_size,
            "C": C,
            "solver": solver,
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
    }


# ─────────────────────────────────────────────
# Plot helpers (return PNG bytes → Kivy Image)
# ─────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor":  "#1e1e2e",
    "axes.facecolor":    "#2a2a3e",
    "axes.edgecolor":    "#555577",
    "axes.labelcolor":   "#cdd6f4",
    "text.color":        "#cdd6f4",
    "xtick.color":       "#cdd6f4",
    "ytick.color":       "#cdd6f4",
    "grid.color":        "#363650",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> bytes:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
            ax=ax, linewidths=0.5,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        data = _fig_to_bytes(fig)
        plt.close(fig)
    return data


def plot_feature_importance(top_features: pd.Series, save_path: str = None) -> bytes:
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.barh(top_features.index, top_features.values, color=colors, edgecolor="none")
        ax.set_xlabel("Coefficient Magnitude", fontsize=11)
        ax.set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x")
        for bar, val in zip(bars, top_features.values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, color="#cdd6f4")
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        data = _fig_to_bytes(fig)
        plt.close(fig)
    return data


def plot_roc_curve(fpr, tpr, roc_auc: float, save_path: str = None) -> bytes:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#89b4fa", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.1f}%)")
        ax.plot([0, 1], [0, 1], color="#585b70", lw=1, linestyle="--", label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#89b4fa")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        data = _fig_to_bytes(fig)
        plt.close(fig)
    return data


def plot_cv_scores(cv_scores: list, save_path: str = None) -> bytes:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
        bar_colors = ["#a6e3a1" if s >= np.mean(cv_scores) else "#f38ba8" for s in cv_scores]
        ax.bar(folds, cv_scores, color=bar_colors, edgecolor="none", width=0.5)
        ax.axhline(np.mean(cv_scores), color="#f9e2af", linestyle="--", lw=1.5,
                   label=f"Mean: {np.mean(cv_scores):.1f}%")
        ax.set_ylim([min(cv_scores) - 3, 100])
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Cross-Validation Scores", fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y")
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        data = _fig_to_bytes(fig)
        plt.close(fig)
    return data
