"""Baseline classifiers, Isolation Forest anomaly score, and hybrid evaluation."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix, accuracy_score)

def train_and_evaluate(X, y, alpha=0.7, seed=42):
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
        random_state=seed, stratify=y)
    estimators = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=seed))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced",
                min_samples_leaf=2, random_state=seed, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed))
        ])
    }
    fitted, probabilities, metrics = {}, {}, []
    for name, pipe in estimators.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probabilities[name] = pipe, p
        pred = (p >= 0.5).astype(int)
        metrics.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
            "PR-AUC": average_precision_score(yte, p),
            "ROC-AUC": roc_auc_score(yte, p),
            "MCC": matthews_corrcoef(yte, pred)})
    imputer = SimpleImputer(strategy="median")
    Xtr_i, Xte_i = imputer.fit_transform(Xtr), imputer.transform(Xte)
    iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=250).fit(Xtr_i)
    raw = -iso.decision_function(Xte_i)
    lo, hi = float(raw.min()), float(raw.max())
    anomaly = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    best_name = max(probabilities, key=lambda n: average_precision_score(yte, probabilities[n]))
    hybrid = alpha * probabilities[best_name] + (1-alpha) * anomaly
    pred = (hybrid >= 0.5).astype(int)
    metrics.append({"Model": f"Hybrid ({best_name})", "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1": f1_score(yte, pred, zero_division=0),
        "PR-AUC": average_precision_score(yte, hybrid),
        "ROC-AUC": roc_auc_score(yte, hybrid),
        "MCC": matthews_corrcoef(yte, pred)})
    return {"models": fitted, "metrics": metrics, "best_name": best_name,
        "hybrid_scores": hybrid, "anomaly_scores": anomaly, "y_test": yte,
        "X_test": Xte, "confusion": confusion_matrix(yte, pred),
        "alpha": alpha, "iso": iso, "imputer": imputer}
