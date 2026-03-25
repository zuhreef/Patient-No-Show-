import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Convert datetime columns
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)

    # Remove clearly invalid ages
    df = df[df["Age"] >= 0].copy()

    # Create lead time feature
    df["lead_days"] = (
        (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / 86400
    ).clip(lower=0)

    # Day-of-week feature
    df["appointment_weekday"] = df["AppointmentDay"].dt.dayofweek

    # Target variable
    df["no_show"] = df["No-show"].map({"Yes": 1, "No": 0})

    # Encode Gender
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

    # One-hot encode neighbourhood
    df = pd.get_dummies(df, columns=["Neighbourhood"], drop_first=True)

    # Features to exclude
    exclude_cols = [
        "PatientId",
        "AppointmentID",
        "ScheduledDay",
        "AppointmentDay",
        "No-show",
        "no_show",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df["no_show"]

    if len(df) < 10:
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test, feature_cols