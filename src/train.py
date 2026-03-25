from pathlib import Path
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")
FEATURES_PATH = Path("artifacts/feature_columns.json")


def train_model(X_train, y_train, X_test, y_test, feature_cols):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    roc_auc = roc_auc_score(y_test, pred_probs)

    metrics = {
        "roc_auc": roc_auc,
        "classification_report": report,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    return model, metrics