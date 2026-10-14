"""Feature definitions, time-based split and model pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Low-cardinality categoricals: one-hot / native categorical.
LOW_CARD = ["property_type_label", "old_new_label", "duration_label", "region"]
# High-cardinality geography: target-encoded (cross-fitted inside the pipeline, so no leakage).
HIGH_CARD = ["postcode_area", "postcode_district", "district", "county", "town_city"]
NUMERIC = ["month_sin", "month_cos"]
FEATURES = LOW_CARD + HIGH_CARD + NUMERIC
TARGET = "log_price"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log-price target and cyclical month features.

    No trend/year feature is used on purpose: tree models cannot extrapolate to
    an unseen year, so seasonality is the only time signal.
    """
    out = df.copy()
    out[TARGET] = np.log(out["price"].astype(float))
    month = out["date_of_transfer"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def time_split(
    df: pd.DataFrame, test_start: str = "2025-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on sales before ``test_start``, test on sales from that date.

    A time-based split mimics real use (predict the future from the past) and
    avoids the optimism of a random split.
    """
    cutoff = pd.Timestamp(test_start)
    train = df[df["date_of_transfer"] < cutoff]
    test = df[df["date_of_transfer"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError(f"Split at {test_start} leaves an empty train or test set")
    return train, test


class MedianByDistrict:
    """Naive baseline: predict the training median log-price of the district."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> MedianByDistrict:
        self.overall_ = float(y.median())
        self.by_district_ = y.groupby(df["district"].to_numpy()).median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["district"].map(self.by_district_).fillna(self.overall_).to_numpy()


def _encoders(scale: bool) -> ColumnTransformer:
    num = [("num", StandardScaler() if scale else "passthrough", NUMERIC)]
    return ColumnTransformer(
        [
            ("low", OneHotEncoder(handle_unknown="ignore", min_frequency=20), LOW_CARD),
            (
                "high",
                TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)),
                HIGH_CARD,
            ),
            *num,
        ]
    )


def build_models() -> dict[str, object]:
    """The candidate models, from naive to strongest."""
    return {
        "Baseline: district median": MedianByDistrict(),
        "Global mean": Pipeline([("x", _encoders(False)), ("m", DummyRegressor())]),
        "Ridge regression": Pipeline([("x", _encoders(True)), ("m", Ridge(alpha=1.0))]),
        "Gradient boosting": Pipeline(
            [
                ("x", _encoders(False)),
                (
                    "m",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.08,
                        max_leaf_nodes=63,
                        early_stopping=True,
                        random_state=0,
                    ),  # fmt: skip
                ),
            ]
        ),
    }
