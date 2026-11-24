import numpy as np
import pandas as pd


def build_features(df):
    features = df.copy()

    features["log_request_rate"] = np.log1p(
        features["requests_per_minute"]
    )

    features["requests_per_endpoint"] = (
        features["requests_per_minute"]
        / features["unique_endpoints"].clip(lower=1)
    )

    features["estimated_errors_per_minute"] = (
        features["requests_per_minute"]
        * features["error_rate"]
    )

    feature_columns = [
        "requests_per_minute",
        "request_interval",
        "error_rate",
        "unique_endpoints",
        "log_request_rate",
        "requests_per_endpoint",
        "estimated_errors_per_minute"
    ]

    X = features[feature_columns]
    y = features["is_abusive"]

    return X, y


if __name__ == "__main__":
    df = pd.read_csv("data/traffic.csv")
    X, y = build_features(df)

    print("Feature engineering completed!")
    print("Feature shape:", X.shape)
    print(X.head())
    print("Missing values:", X.isnull().sum().sum())