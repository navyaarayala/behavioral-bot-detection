import pandas as pd
from src.generate_data import generate_traffic
from src.features import build_features


def test_dataset_size():
    df = generate_traffic(n_samples=100)
    assert len(df) == 100


def test_valid_categories():
    df = generate_traffic(n_samples=100)
    expected = {
        "human",
        "legitimate_automation",
        "abusive_bot"
    }
    assert set(df["category"]).issubset(expected)


def test_binary_labels():
    df = generate_traffic(n_samples=100)
    assert set(df["is_abusive"]).issubset({0, 1})


def test_feature_dimensions():
    df = generate_traffic(n_samples=100)
    X, y = build_features(df)
    assert X.shape == (100, 7)
    assert len(y) == 100


def test_no_missing_values():
    df = generate_traffic(n_samples=100)
    X, y = build_features(df)
    assert X.isnull().sum().sum() == 0


def test_no_label_leakage():
    df = generate_traffic(n_samples=100)
    X, y = build_features(df)
    assert "is_abusive" not in X.columns
    assert "category" not in X.columns


def test_reproducibility():
    first = generate_traffic(100, seed=42)
    second = generate_traffic(100, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_label_consistency():
    df = generate_traffic(n_samples=100)
    expected = (df["category"] == "abusive_bot").astype(int)
    assert (df["is_abusive"] == expected).all()