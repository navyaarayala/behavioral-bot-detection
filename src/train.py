from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.features import build_features


def train_models():
    df = pd.read_csv("data/traffic.csv")
    X, y = build_features(df)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=0.5,
        random_state=42,
        stratify=y_holdout
    )

    # Supervised classification
    classifier = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(X_train, y_train)

    # Anomaly detection trained on legitimate traffic
    anomaly_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    anomaly_model.fit(X_train[y_train == 0])

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        classifier,
        "models/random_forest.joblib"
    )

    joblib.dump(
        anomaly_model,
        "models/isolation_forest.joblib"
    )

    joblib.dump(
        (X_val, X_test, y_val, y_test),
        "models/evaluation_data.joblib"
    )

    print("Models trained successfully!")

    print("\nRandom Forest - Validation Results:")
    print(classification_report(
        y_val, classifier.predict(X_val)
    ))

    anomaly_predictions = (
        anomaly_model.predict(X_val) == -1
    ).astype(int)

    print("\nIsolation Forest - Validation Results:")
    print(classification_report(
        y_val, anomaly_predictions
    ))


if __name__ == "__main__":
    train_models()