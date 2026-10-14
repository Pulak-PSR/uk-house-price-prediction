"""Load the cleaned Price Paid Data produced by uk-property-price-pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REGION_LOOKUP = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "postcode_area_region.csv"
)
COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode_district",
    "property_type_label",
    "old_new_label",
    "duration_label",
    "district",
    "county",
    "town_city",
    "is_outlier",
]


def add_region(df: pd.DataFrame, lookup_path: Path = REGION_LOOKUP) -> pd.DataFrame:
    """Add ``postcode_area`` and ``region`` columns from the postcode-district."""
    lookup = pd.read_csv(lookup_path, dtype=str)
    mapping = dict(zip(lookup["postcode_area"], lookup["region"], strict=True))
    area = df["postcode_district"].astype("string").str.extract(r"^([A-Z]+)", expand=False)
    out = df.copy()
    out["postcode_area"] = area.fillna("UNKNOWN").astype(str)
    out["region"] = area.map(mapping).fillna("Unknown").astype(str)
    return out


def load_cleaned(path: Path, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read cleaned.parquet, drop flagged outliers and rows without a postcode.

    Outliers are removed for modelling only (the pipeline keeps them, flagged).
    """
    df = pd.read_parquet(path, columns=COLUMNS)
    n0 = len(df)
    df = df[~df["is_outlier"] & df["postcode_district"].notna()].drop(columns="is_outlier")
    logger.info(
        "Loaded %d rows, kept %d after dropping outliers and missing postcodes", n0, len(df)
    )
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
        logger.info("Sampled %d rows", len(df))
    df = df.reset_index(drop=True)
    for col in ["postcode_district", "property_type_label", "old_new_label", "duration_label",
                "district", "county", "town_city"]:  # fmt: skip
        df[col] = df[col].astype(str)
    return add_region(df)
