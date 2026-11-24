import numpy as np
import pandas as pd
from pathlib import Path


def generate_traffic(n_samples=10000, seed=42):
    rng = np.random.default_rng(seed)

    profiles = {
        "human": (25, 12, 0.30, 0.15),
        "legitimate_automation": (100, 40, 0.10, 0.10),
        "abusive_bot": (160, 65, 0.08, 0.45),
    }

    traffic = []

    for _ in range(n_samples):
        category = rng.choice(
            list(profiles),
            p=[0.65, 0.20, 0.15]
        )

        rate, spread, interval, error = profiles[category]

        traffic.append({
            "requests_per_minute": max(
                1, int(rng.normal(rate, spread))
            ),
            "request_interval": rng.exponential(interval),
            "error_rate": np.clip(
                rng.normal(error, 0.15), 0, 1
            ),
            "unique_endpoints": rng.integers(1, 30),
            "category": category,
            "is_abusive": int(category == "abusive_bot")
        })

    return pd.DataFrame(traffic)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    df = generate_traffic()
    df.to_csv("data/traffic.csv", index=False)

    print("Dataset generated successfully!")
    print(df.head())
    print(df["category"].value_counts())