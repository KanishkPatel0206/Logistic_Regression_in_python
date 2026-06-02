# 🔬 Breast Cancer Classification — Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/breast-cancer-logistic-regression/blob/main/notebooks/logistic_regression.ipynb)

> **Can a model learn to distinguish a benign tumour from a malignant one?**  
> This project builds a Logistic Regression classifier on the Breast Cancer Wisconsin dataset — achieving **~97% accuracy** on the test set.

---

## 📌 Overview

| Item | Detail |
|---|---|
| **Dataset** | [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| **Task** | Binary Classification (Benign vs Malignant) |
| **Algorithm** | Logistic Regression |
| **Test Accuracy** | ~97% |
| **Features** | 30 numeric features from digitized cell nuclei images |
| **Samples** | 569 (357 benign, 212 malignant) |

---

## 🧠 What You'll Learn

- Loading and exploring a real-world medical dataset with `sklearn`
- Preprocessing with `StandardScaler` for better model convergence
- Training and evaluating a Logistic Regression classifier
- Interpreting key classification metrics (Accuracy, Precision, Recall, F1)
- Visualizing a Confusion Matrix and feature importance

---

## 📁 Project Structure

```
breast-cancer-logistic-regression/
├── src/
│   └── logistic_regression.py      # Main training & evaluation script
├── notebooks/
│   └── logistic_regression.ipynb   # Step-by-step walkthrough (Google Colab)
├── results/
│   ├── confusion_matrix.png        # Auto-generated on first run
│   └── feature_importance.png      # Auto-generated on first run
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-logistic-regression.git
cd breast-cancer-logistic-regression
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the script

```bash
python src/logistic_regression.py
```

Plots will be saved to the `results/` folder automatically.

> **Prefer notebooks?** Open `notebooks/logistic_regression.ipynb` in Jupyter or click the **Open in Colab** badge above.

---

## ⚙️ How It Works

```
Raw Data  →  Split (80/20)  →  StandardScaler  →  LogisticRegression  →  Evaluate  →  Visualize
```

| Step | What Happens |
|---|---|
| **1. Load** | Breast Cancer Wisconsin dataset loaded via `sklearn.datasets` |
| **2. Split** | 80% training / 20% test with `train_test_split` |
| **3. Scale** | Features normalized using `StandardScaler` (zero mean, unit variance) |
| **4. Train** | `LogisticRegression` fitted on training set |
| **5. Evaluate** | Accuracy, Precision, Recall, F1, and Confusion Matrix computed |
| **6. Visualize** | Plots saved to `results/` |

---

## 📊 Results

### Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~97% |
| **Precision** | ~97% |
| **Recall** | ~98% |
| **F1 Score** | ~97% |

> *Results may vary slightly across runs due to random train/test splits.*

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Feature Importance (Model Coefficients)

![Feature Importance](results/feature_importance.png)

---

## 📖 Metrics Explained

| Metric | Formula | Why It Matters Here |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Avoid false cancer alarms |
| **Recall** | TP / (TP + FN) | Critical — missing a malignant case is costly |
| **F1 Score** | 2 × (P × R) / (P + R) | Balances Precision and Recall |

> In medical diagnostics, **Recall** is often prioritised over Precision — a false negative (missed malignancy) is more dangerous than a false positive.

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

Made with ❤️ as part of a **Hands-On Machine Learning** series.  
If this helped you, consider giving it a ⭐ — it helps others find it too!

**Contributions welcome** — feel free to open an issue or PR if you'd like to extend this with SVM, Random Forest, or cross-validation comparisons.
