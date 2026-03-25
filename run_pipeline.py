from src.ingest import load_raw_data
from src.validate import validate_columns, validate_basic_rules
from src.preprocess import preprocess
from src.train import train_model


def main():
    df = load_raw_data()
    validate_columns(df)
    validate_basic_rules(df)

    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)
    _, metrics = train_model(X_train, y_train, X_test, y_test, feature_cols)

    print("Training complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Classification report saved to artifacts/metrics.json")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()