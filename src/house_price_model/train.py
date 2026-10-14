"""Train, evaluate and compare the models. Run: python -m house_price_model.train."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import load_cleaned
from .evaluate import (
    plot_error_by_region,
    plot_importance,
    plot_model_comparison,
    plot_pred_vs_actual,
    regression_metrics,
)
from .features import FEATURES, TARGET, add_features, build_models, time_split

logger = logging.getLogger("house_price_model")
ROOT = Path(__file__).resolve().parents[2]


def run(data: Path, sample: int | None, out_dir: Path, test_start: str) -> pd.DataFrame:
    """Fit all models on the training period and score them on the test period."""
    df = add_features(load_cleaned(data, sample=sample))
    train, test = time_split(df, test_start)
    logger.info("Train: %d rows (to %s), test: %d rows", len(train), test_start, len(test))

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, preds, fitted = {}, {}, {}
    for name, model in build_models().items():
        logger.info("Fitting %s", name)
        if name.startswith("Baseline"):
            model.fit(train, train[TARGET])
            pred = model.predict(test)
        else:
            model.fit(train[FEATURES], train[TARGET])
            pred = model.predict(test[FEATURES])
        rows[name], preds[name], fitted[name] = regression_metrics(test[TARGET], pred), pred, model
        logger.info("  %s", {k: round(v, 2) for k, v in rows[name].items()})

    results = pd.DataFrame(rows).T
    best = "Gradient boosting"
    plot_model_comparison(results, fig_dir / "model_comparison.png")
    plot_pred_vs_actual(test[TARGET].to_numpy(), preds[best], fig_dir / "predicted_vs_actual.png")
    by_region = plot_error_by_region(test, preds[best], fig_dir / "error_by_region.png")

    logger.info("Permutation importance on a test subsample")
    sub = test.sample(n=min(20_000, len(test)), random_state=0)
    imp = permutation_importance(
        fitted[best], sub[FEATURES], sub[TARGET], n_repeats=3, random_state=0,
        scoring="neg_mean_squared_error", n_jobs=1,
    )  # fmt: skip
    importance = pd.Series(imp.importances_mean, index=FEATURES)
    plot_importance(importance, fig_dir / "feature_importance.png")

    payload = {
        "rows_train": len(train),
        "rows_test": len(test),
        "test_start": test_start,
        "models": results.round(4).to_dict(orient="index"),
        "error_by_region": by_region.round(2).to_dict(orient="index"),
        "importance": importance.round(5).to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(fitted[best], model_dir / "gradient_boosting.joblib")
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Train UK house price models")
    p.add_argument("--data", type=Path, required=True, help="cleaned.parquet from the pipeline")
    p.add_argument("--sample", type=int, help="random sample of N rows for a quick run")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--out", type=Path, default=ROOT / "reports")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run(args.data, args.sample, args.out, args.test_start)
    print(results.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
