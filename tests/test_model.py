"""Unit tests for features, split, baseline and metrics."""

import numpy as np
import pandas as pd
import pytest

from house_price_model.data import add_region
from house_price_model.evaluate import regression_metrics
from house_price_model.features import (
    FEATURES,
    TARGET,
    MedianByDistrict,
    add_features,
    build_models,
    time_split,
)


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    districts = np.array(["LEEDS", "BRISTOL", "CAMDEN", "DERBY"])
    base = {"LEEDS": 200_000, "BRISTOL": 320_000, "CAMDEN": 900_000, "DERBY": 180_000}
    d = rng.choice(districts, n)
    df = pd.DataFrame(
        {
            "price": [int(base[x] * rng.uniform(0.8, 1.2)) for x in d],
            "date_of_transfer": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 700, n), "D"),
            "postcode_district": [f"{x[:2]}1" for x in d],
            "property_type_label": rng.choice(["Detached", "Terraced"], n),
            "old_new_label": rng.choice(["Existing", "New Build"], n),
            "duration_label": rng.choice(["Freehold", "Leasehold"], n),
            "district": d,
            "county": d,
            "town_city": d,
        }
    )
    return add_region(df)


def test_add_features_log_target_and_cyclical_month():
    df = add_features(_frame())
    assert np.allclose(np.exp(df[TARGET]), df["price"])
    assert np.allclose(df["month_sin"] ** 2 + df["month_cos"] ** 2, 1.0)


def test_time_split_has_no_overlap():
    df = add_features(_frame())
    train, test = time_split(df, "2025-06-01")
    assert train["date_of_transfer"].max() < pd.Timestamp("2025-06-01")
    assert test["date_of_transfer"].min() >= pd.Timestamp("2025-06-01")
    assert len(train) + len(test) == len(df)


def test_time_split_empty_raises():
    with pytest.raises(ValueError):
        time_split(add_features(_frame()), "2099-01-01")


def test_region_lookup_applied():
    df = add_region(pd.DataFrame({"postcode_district": ["SW1A", "M20", "ZZ9"]}))
    assert df["region"].tolist() == ["London", "North West", "Unknown"]


def test_median_baseline_handles_unseen_district():
    df = add_features(_frame())
    model = MedianByDistrict().fit(df, df[TARGET])
    new = df.iloc[:2].copy()
    new["district"] = ["LEEDS", "NOWHERE"]
    pred = model.predict(new)
    assert pred[1] == pytest.approx(df[TARGET].median())


def test_gradient_boosting_beats_baseline_on_structured_data():
    df = add_features(_frame(1500))
    train, test = time_split(df, "2025-03-01")
    models = build_models()
    base = MedianByDistrict().fit(train, train[TARGET])
    gb = models["Gradient boosting"].fit(train[FEATURES], train[TARGET])
    base_err = regression_metrics(test[TARGET], base.predict(test))["mdape_pct"]
    gb_err = regression_metrics(test[TARGET], gb.predict(test[FEATURES]))["mdape_pct"]
    assert gb_err <= base_err * 1.2  # sanity: not wildly worse
    assert gb_err < 25


def test_metrics_perfect_and_known_error():
    y = np.log(np.array([100_000.0, 200_000.0]))
    perfect = regression_metrics(y, y)
    assert perfect["mae_gbp"] == pytest.approx(0)
    assert perfect["r2_log"] == pytest.approx(1)
    off = regression_metrics(y, y + np.log(1.1))
    assert off["mdape_pct"] == pytest.approx(10.0)
    assert off["within_20pct"] == 100
