from pathlib import Path
import pandas as pd
import urllib.request

RAW_PATH = Path("data/raw/appointments.csv")

# Public dataset URL (GitHub mirror)
DATA_URL = "https://raw.githubusercontent.com/joniarroba/noshowappointments/master/KaggleV2-May-2016.csv"


def download_data():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    print("Download complete.")


def load_raw_data() -> pd.DataFrame:
    """
    Load dataset. If not found, download automatically.
    """
    if not RAW_PATH.exists():
        download_data()

    df = pd.read_csv(RAW_PATH)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded raw data successfully.")
    print(f"Shape: {df.shape}")
    print(df.head())