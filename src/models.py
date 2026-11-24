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


def train_and_evaluate(X, y, alpha=0.7, seed=42):
    """Train models, select the hybrid base model on validation data, and test once."""
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")

    # Keep the final test partition separate from model selection.
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

    # Select the hybrid's supervised model using validation PR-AUC only.
    validation_scores = {}
    for name, pipe in make_estimators().items():
        pipe.fit(Xfit, yfit)
        validation_probability = pipe.predict_proba(Xval)[:, 1]
        validation_scores[name] = average_precision_score(yval, validation_probability)

    best_name = max(validation_scores, key=validation_scores.get)

    # Refit each baseline model on the complete training partition.
    fitted, probabilities, metrics = {}, {}, []
    for name, pipe in make_estimators().items():
        pipe.fit(Xtr, ytr)
        probability = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probabilities[name] = pipe, probability
        prediction = (probability >= 0.5).astype(int)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, prediction),
            "Precision": precision_score(yte, prediction, zero_division=0),
            "Recall": recall_score(yte, prediction, zero_division=0),
            "F1": f1_score(yte, prediction, zero_division=0),
            "PR-AUC": average_precision_score(yte, probability),
            "ROC-AUC": roc_auc_score(yte, probability),
            "MCC": matthews_corrcoef(yte, prediction)
        })

    # Fit the anomaly detector and imputer on training data only.
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
    metrics.append({
        "Model": f"Hybrid ({best_name})",
        "Accuracy": accuracy_score(yte, hybrid_prediction),
        "Precision": precision_score(yte, hybrid_prediction, zero_division=0),
        "Recall": recall_score(yte, hybrid_prediction, zero_division=0),
        "F1": f1_score(yte, hybrid_prediction, zero_division=0),
        "PR-AUC": average_precision_score(yte, hybrid),
        "ROC-AUC": roc_auc_score(yte, hybrid),
        "MCC": matthews_corrcoef(yte, hybrid_prediction)
    })

    return {
        "models": fitted,
        "metrics": metrics,
        "best_name": best_name,
        "hybrid_scores": hybrid,
        "anomaly_scores": anomaly,
        "y_test": yte,
        "X_test": Xte,
        "confusion": confusion_matrix(yte, hybrid_prediction, labels=[0, 1]),
        "alpha": alpha,
        "iso": iso,
        "imputer": anomaly_imputer,
        "validation_scores": validation_scores
    }
