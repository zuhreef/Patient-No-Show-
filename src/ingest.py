from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/appointments.csv")
SAMPLE_PATH = Path("data/reference/appointments_sample.csv")


def load_raw_data() -> pd.DataFrame:
    """
    Load the full local dataset when available.
    If it's missing, fall back to the committed sample dataset.
    """
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    if SAMPLE_PATH.exists():
        print("Full dataset not found. Using sample dataset.")
        return pd.read_csv(SAMPLE_PATH)

    raise FileNotFoundError(
        f"No dataset found. Expected either {RAW_PATH} or {SAMPLE_PATH}."
    )


if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded data successfully.")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())