REQUIRED_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
]


def validate_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_basic_rules(df):
    if df.empty:
        raise ValueError("Input dataframe is empty.")