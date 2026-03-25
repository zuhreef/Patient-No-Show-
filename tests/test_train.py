from pathlib import Path
import pandas as pd

from src.preprocess import preprocess
from src.train import train_model


def test_training_creates_artifacts():
    df = pd.DataFrame({
        "PatientId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "AppointmentID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
        "ScheduledDay": [
            "2016-04-29T18:38:08Z", "2016-04-29T16:08:27Z",
            "2016-04-29T16:19:04Z", "2016-04-29T17:29:31Z",
            "2016-04-29T18:00:00Z", "2016-04-29T18:10:00Z",
            "2016-04-29T18:20:00Z", "2016-04-29T18:30:00Z",
            "2016-04-29T18:40:00Z", "2016-04-29T18:50:00Z",
        ],
        "AppointmentDay": [
            "2016-05-02T00:00:00Z", "2016-05-03T00:00:00Z",
            "2016-05-04T00:00:00Z", "2016-05-05T00:00:00Z",
            "2016-05-06T00:00:00Z", "2016-05-07T00:00:00Z",
            "2016-05-08T00:00:00Z", "2016-05-09T00:00:00Z",
            "2016-05-10T00:00:00Z", "2016-05-11T00:00:00Z",
        ],
        "Age": [29, 45, 52, 36, 41, 33, 50, 27, 39, 60],
        "Neighbourhood": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
        "Scholarship": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Hipertension": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Diabetes": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "Alcoholism": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        "Handcap": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SMS_received": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "No-show": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    })

    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)
    _, metrics = train_model(X_train, y_train, X_test, y_test, feature_cols)

    assert "roc_auc" in metrics
    assert Path("models/model.pkl").exists()
    assert Path("artifacts/metrics.json").exists()
    assert Path("artifacts/feature_columns.json").exists()