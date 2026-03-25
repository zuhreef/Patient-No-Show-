from src.ingest import load_raw_data
from src.validate import validate_columns, validate_basic_rules
from src.preprocess import preprocess


def main():
    df = load_raw_data()
    validate_columns(df)
    validate_basic_rules(df)

    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print("First 10 features:")
    print(feature_cols[:10])


if __name__ == "__main__":
    main()