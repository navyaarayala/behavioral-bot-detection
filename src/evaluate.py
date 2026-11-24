from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)


def evaluate():
    Path("results").mkdir(exist_ok=True)

    model = joblib.load(
        "models/random_forest.joblib"
    )

    X_val, X_test, y_val, y_test = joblib.load(
        "models/evaluation_data.joblib"
    )

    val_scores = model.predict_proba(X_val)[:, 1]

    results = []

    for threshold in np.arange(0.1, 1.0, 0.05):
        predictions = (val_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "false_positive_rate": fp / (fp + tn)
        })

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        "results/threshold_analysis.csv",
        index=False
    )

    # Select threshold with validation FPR below 2%
    candidates = metrics[
        metrics["false_positive_rate"] <= 0.02
    ]

    if candidates.empty:
        best = metrics.sort_values(
            ["false_positive_rate", "f1"],
            ascending=[True, False]
        ).iloc[0]
    else:
        best = candidates.loc[candidates["f1"].idxmax()]

    threshold = best["threshold"]

    print("Selected threshold:", threshold)
    print("\nValidation metrics:")
    print(best)

    test_scores = model.predict_proba(X_test)[:, 1]
    predictions = (test_scores >= threshold).astype(int)

    print("\nFinal Test Results:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Average Precision:",
          average_precision_score(y_test, test_scores))

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("Test False Positive Rate:", fp / (fp + tn))

    # Evaluate legitimate automation separately
    traffic = pd.read_csv("data/traffic.csv")
    categories = traffic.loc[X_test.index, "category"]

    for category in ["human", "legitimate_automation"]:
        mask = categories.to_numpy() == category
        print(category, "false positive rate:",
              predictions[mask].mean())

    PrecisionRecallDisplay.from_predictions(
        y_test, test_scores
    )
    plt.title("Bot Detection - Precision Recall Curve")
    plt.tight_layout()
    plt.savefig("results/precision_recall.png")
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Legitimate", "Abusive"]
    )
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate()