# -*- coding: utf-8 -*-
"""
Breast Cancer Classification using Logistic Regression
Dataset: Breast Cancer Wisconsin (Diagnostic) from sklearn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and return the Breast Cancer dataset as a DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 1 = malignant, 0 = benign
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def preprocess(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data into train/test sets and apply StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """Train and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: pd.Series) -> None:
    """Print accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)

    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return y_pred


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Plot and save the confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — Breast Cancer Classification")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {save_path}")


def plot_feature_importance(
    model: LogisticRegression,
    feature_names: list,
    top_n: int = 10,
    save_path: str = "results/feature_importance.png",
) -> None:
    """Plot and save the top N most important features by coefficient magnitude."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    coefs = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    top_features = coefs.nlargest(top_n).sort_values()

    plt.figure(figsize=(8, 5))
    top_features.plot(kind="barh", color="steelblue")
    plt.xlabel("Coefficient Magnitude")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")


def main():
    print("Loading data...")
    X, y = load_data()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = evaluate_model(model, X_test, y_test)

    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, list(X.columns))

    print("\nDone!")


if __name__ == "__main__":
    main()
