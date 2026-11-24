"""Baseline classifiers, Isolation Forest anomaly score, and hybrid evaluation."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix, accuracy_score
)


def _classification_summary(y_true, prediction, probability):
    """Calculate all threshold-based metrics and the matrix from one prediction."""
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tn + fp + fn + tp
    return {
        "Accuracy": (tn + tp) / total if total else 0.0,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": average_precision_score(y_true, probability),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "MCC": matthews_corrcoef(y_true, prediction),
    }, matrix


def train_and_evaluate(X, y, alpha=0.7, seed=42):
    """Train models, select the hybrid base model on validation data, and test once."""
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    Xfit, Xval, yfit, yval = train_test_split(
        Xtr, ytr, test_size=0.20, random_state=seed, stratify=ytr
    )

    def make_estimators():
        return {
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced", max_iter=1500, random_state=seed
                ))
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=250, class_weight="balanced",
                    min_samples_leaf=2, random_state=seed, n_jobs=-1
                ))
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.08, random_state=seed
                ))
            ])
        }

    validation_scores = {}
    for name, pipe in make_estimators().items():
        pipe.fit(Xfit, yfit)
        validation_probability = pipe.predict_proba(Xval)[:, 1]
        validation_scores[name] = average_precision_score(yval, validation_probability)

    best_name = max(validation_scores, key=validation_scores.get)

    fitted, probabilities, metrics = {}, {}, []
    for name, pipe in make_estimators().items():
        pipe.fit(Xtr, ytr)
        probability = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probabilities[name] = pipe, probability
        prediction = (probability >= 0.5).astype(int)
        summary, _ = _classification_summary(yte, prediction, probability)
        metrics.append({"Model": name, **summary})

    anomaly_imputer = SimpleImputer(strategy="median")
    Xtr_i = anomaly_imputer.fit_transform(Xtr)
    Xte_i = anomaly_imputer.transform(Xte)
    iso = IsolationForest(
        contamination="auto", random_state=seed, n_estimators=250
    ).fit(Xtr_i)

    raw_train_anomaly = -iso.decision_function(Xtr_i)
    raw_test_anomaly = -iso.decision_function(Xte_i)
    low, high = float(raw_train_anomaly.min()), float(raw_train_anomaly.max())
    if high > low:
        anomaly = np.clip((raw_test_anomaly - low) / (high - low), 0.0, 1.0)
    else:
        anomaly = np.zeros_like(raw_test_anomaly)

    hybrid = alpha * probabilities[best_name] + (1.0 - alpha) * anomaly
    hybrid_prediction = (hybrid >= 0.5).astype(int)
    hybrid_summary, hybrid_confusion = _classification_summary(
        yte, hybrid_prediction, hybrid
    )
    metrics.append({"Model": f"Hybrid ({best_name})", **hybrid_summary})

    return {
        "models": fitted,
        "metrics": metrics,
        "best_name": best_name,
        "hybrid_scores": hybrid,
        "hybrid_prediction": hybrid_prediction,
        "anomaly_scores": anomaly,
        "y_test": yte,
        "X_test": Xte,
        "confusion": hybrid_confusion,
        "alpha": alpha,
        "iso": iso,
        "imputer": anomaly_imputer,
        "validation_scores": validation_scores
    }
