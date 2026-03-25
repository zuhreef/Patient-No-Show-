from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/appointments.csv")


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw appointment dataset from the local data/raw directory.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. "
            "Download the dataset and place it in data/raw/appointments.csv"
        )

    df = pd.read_csv(RAW_PATH)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded raw data successfully.")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())