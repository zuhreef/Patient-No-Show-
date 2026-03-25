import pandas as pd
from src.preprocess import preprocess


def test_preprocess_runs():
    df = pd.DataFrame({
        "PatientId": [1, 2, 3, 4],
        "AppointmentID": [101, 102, 103, 104],
        "Gender": ["F", "M", "F", "M"],
        "ScheduledDay": [
            "2016-04-29T18:38:08Z",
            "2016-04-29T16:08:27Z",
            "2016-04-29T16:19:04Z",
            "2016-04-29T17:29:31Z",
        ],
        "AppointmentDay": [
            "2016-05-02T00:00:00Z",
            "2016-05-03T00:00:00Z",
            "2016-05-04T00:00:00Z",
            "2016-05-05T00:00:00Z",
        ],
        "Age": [29, 45, 52, 36],
        "Neighbourhood": ["A", "B", "A", "C"],
        "Scholarship": [0, 1, 0, 1],
        "Hipertension": [0, 1, 0, 1],
        "Diabetes": [0, 0, 1, 1],
        "Alcoholism": [0, 0, 0, 1],
        "Handcap": [0, 0, 0, 0],
        "SMS_received": [1, 0, 1, 0],
        "No-show": ["No", "Yes", "No", "Yes"],
    })

    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)

    assert len(feature_cols) > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) > 0
    assert len(y_test) > 0