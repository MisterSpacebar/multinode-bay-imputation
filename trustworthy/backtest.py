"""Leakage-aware observed-only daily hypoxia forecasting benchmarks.

All predictions use information available at the END of issue_date. Each
future target date must be strictly later than the issue date; training
targets must end strictly BEFORE the issue date. Persistence comparisons
are paired on exactly the same target cases as each candidate model.

This is a retrospective benchmark, not an operational early warning model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 7)
TEST_MONTHS = ("2025-09", "2026-09")  # API compatibility; CLI defaults to all

DO_FEATURES = ("hours_lt4", "odo_mean", "odo_min", "odo_range")
PHYSICAL_FEATURES = DO_FEATURES + ("temp_mean", "sal_mean")
FEATURES = list(PHYSICAL_FEATURES + ("predawn_min", "trail7"))

# Models have distinct, explicitly documented predictor requirements.
# The issue-day DO value is mandatory for every model so persistence is a
# meaningful paired reference. Train mean also needs historical labels.
MODEL_FEATURES = {
    "persistence": ("hours_lt4",),
    "trail7": ("hours_lt4", "trail7"),
    "train_mean": ("hours_lt4",),
    "ridge_do": DO_FEATURES,
    "ridge_physical": PHYSICAL_FEATURES,
    "ridge": tuple(FEATURES),  # historical full-feature Ridge name
}
FITTED_MODELS = frozenset(("train_mean", "ridge_do", "ridge_physical", "ridge"))
PREDICTION_COLUMNS = (
    "node", "issue_date", "target_date", "test_month", "horizon_days",
    "deployment_id", "model", "n_train", "actual_hours_lt4",
    "predicted_hours_lt4", "persistence_prediction", "absolute_error",
    "squared_error",
)


def prepare(frame: pd.DataFrame, node: str, horizon: int) -> pd.DataFrame:
    """Build complete calendar-day grid; never fill missing targets."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one full day")
    d = frame.loc[frame["node"] == node].copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    d = d.sort_values("date").set_index("date")
    if d.index.has_duplicates:
        raise ValueError(f"{node}: duplicate daily dates")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    for name in set(FEATURES) | {"hours_lt4"}:
        if name not in d:
            d[name] = np.nan
        d[name] = pd.to_numeric(d[name], errors="coerce")
    d["trail7"] = d["hours_lt4"].rolling(7, min_periods=4).mean()
    d["target"] = d["hours_lt4"].shift(-horizon)
    d["target_date"] = d.index + pd.Timedelta(days=horizon)
    d["deployment_id"] = d.get(
        "deployment_id", pd.Series("not_provided", index=d.index)
    ).fillna("not_provided")
    d["source_verified"] = d.get(
        "source_verified", pd.Series(False, index=d.index)
    ).astype("string").str.lower().eq("true").fillna(False)
    d["same_epoch"] = (
        d["source_verified"]
        & d["source_verified"].shift(-horizon).fillna(False)
        & d["deployment_id"].eq(d["deployment_id"].shift(-horizon))
    )
    return d


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, columns,
                  alpha: float = 10.0) -> float:
    """Fixed-penalty Ridge; train-only standardization."""
    x = train[list(columns)].to_numpy(float)
    y = train["target"].to_numpy(float)
    q = test[list(columns)].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x, q = (x - mean) / sd, (q - mean) / sd
    ym = y.mean()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym))
    return float(np.clip((q @ w + ym)[0], 0, 24))


def _feature_mask(d: pd.DataFrame, model: str, require_verified: bool):
    valid = d[list(MODEL_FEATURES[model])].notna().all(axis=1)
    if require_verified:
        valid &= d["same_epoch"].fillna(False)
    return valid


def _training(d, issue_date, model, require_verified):
    mask = (d["target"].notna()
            & _feature_mask(d, model, require_verified)
            & (d["target_date"] < issue_date))
    return d.loc[mask]


def _months(daily: pd.DataFrame, test_months):
    if test_months is None:
        return sorted(pd.to_datetime(daily["date"]).dt.to_period("M")
                      .astype(str).unique())
    return [str(x) for x in test_months]


def eligible_forecasts(daily: pd.DataFrame, nodes=None, horizons=HORIZONS,
                       test_months=TEST_MONTHS, require_verified=False,
                       min_train=40):
    """Return (predictions, diagnostic rows), sharing eligibility rules.

    Diagnostic rows include *every model and potential target day* in each
    requested month intersecting station coverage, even when not predictable.
    The exclusive reason labels are used for reproducible exclusion counts.
    """
    if min_train < 1:
        raise ValueError("min_train must be positive")
    months = _months(daily, test_months)
    nodes = list(nodes) if nodes is not None else sorted(daily["node"].unique())
    predictions, diagnostics = [], []
    for node in nodes:
        if not (daily["node"] == node).any():
            continue
        for h in horizons:
            d = prepare(daily, node, h)
            if d.empty:
                continue
            for month in months:
                period = pd.Period(month, freq="M")
                # Count only target dates falling inside the station's own
                # first and last dates; do not invent pre/post deployment days.
                target_days = pd.date_range(
                    max(period.start_time, d.index.min()),
                    min(period.end_time.normalize(), d.index.max()),
                    freq="D",
                )
                for target_day in target_days:
                    issue = target_day - pd.Timedelta(days=h)
                    for model in MODEL_FEATURES:
                        reason = None
                        train = None
                        issue_row = None
                        if not np.isfinite(d.at[target_day, "hours_lt4"]):
                            reason = "missing_target"
                        elif issue not in d.index:
                            reason = "no_issue_history"
                        else:
                            issue_row = d.loc[[issue]]
                            if not np.isfinite(d.at[issue, "hours_lt4"]):
                                reason = "missing_issue_do"
                            elif require_verified and not bool(d.at[issue, "same_epoch"]):
                                reason = "unverified_epoch"
                            elif not _feature_mask(d.loc[[issue]], model, require_verified).iloc[0]:
                                reason = "missing_model_features"
                            elif model in FITTED_MODELS:
                                train = _training(d, issue, model, require_verified)
                                if len(train) < min_train:
                                    reason = "insufficient_training"
                        diagnostics.append({
                            "node": node, "target_date": target_day.date().isoformat(),
                            "issue_date": issue.date().isoformat(),
                            "test_month": month, "horizon_days": h,
                            "model": model, "reason": reason or "evaluable",
                            "n_train": len(train) if train is not None else 0,
                        })
                        if reason is not None:
                            continue
                        actual = float(d.at[target_day, "hours_lt4"])
                        baseline = float(d.at[issue, "hours_lt4"])
                        if model == "persistence":
                            prediction = baseline
                        elif model == "trail7":
                            prediction = float(d.at[issue, "trail7"])
                        elif model == "train_mean":
                            prediction = float(train["target"].mean())
                        else:
                            prediction = ridge_predict(
                                train, issue_row, MODEL_FEATURES[model])
                        predictions.append({
                            "node": node,
                            "issue_date": issue.date().isoformat(),
                            "target_date": target_day.date().isoformat(),
                            "test_month": month, "horizon_days": h,
                            "deployment_id": d.at[issue, "deployment_id"],
                            "model": model,
                            "n_train": len(train) if train is not None else 0,
                            "actual_hours_lt4": actual,
                            "predicted_hours_lt4": prediction,
                            "persistence_prediction": baseline,
                            "absolute_error": abs(actual - prediction),
                            "squared_error": (actual - prediction) ** 2,
                        })
    return (pd.DataFrame(predictions, columns=PREDICTION_COLUMNS),
            pd.DataFrame(diagnostics, columns=[
                "node", "target_date", "issue_date", "test_month",
                "horizon_days", "model", "reason", "n_train",
            ]))


def evaluate(daily: pd.DataFrame, nodes=None, horizons=HORIZONS,
             test_months=TEST_MONTHS, require_verified=False, min_train=40):
    """Backward-compatible predictions-only API."""
    return eligible_forecasts(
        daily, nodes, horizons, test_months, require_verified, min_train
    )[0]


SUMMARY_COLUMNS = (
    "node", "test_month", "horizon_days", "model", "n_test",
    "n_hypoxic_target_days", "mae_hours", "rmse_hours",
    "hypoxic_day_mae_hours", "paired_persistence_mae",
    "paired_persistence_rmse", "rmse_skill_vs_persistence",
)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use the SAME evaluated cases for model and persistence errors."""
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    group = ["node", "test_month", "horizon_days", "model"]
    for (node, month, horizon, model), grp in predictions.groupby(group):
        y = grp["actual_hours_lt4"].to_numpy(float)
        p = grp["predicted_hours_lt4"].to_numpy(float)
        base = grp["persistence_prediction"].to_numpy(float)
        pos = y > 0
        mse = float(np.mean((y - p) ** 2))
        base_mse = float(np.mean((y - base) ** 2))
        rows.append({
            "node": node, "test_month": month,
            "horizon_days": int(horizon), "model": model,
            "n_test": len(y), "n_hypoxic_target_days": int(pos.sum()),
            "mae_hours": float(np.mean(np.abs(y - p))),
            "rmse_hours": float(np.sqrt(mse)),
            "hypoxic_day_mae_hours": (
                float(np.mean(np.abs(y[pos] - p[pos]))) if pos.any() else np.nan),
            "paired_persistence_mae": float(np.mean(np.abs(y - base))),
            "paired_persistence_rmse": float(np.sqrt(base_mse)),
            "rmse_skill_vs_persistence": (
                1.0 - mse / base_mse if base_mse > 0 else np.nan),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Observed-only rolling-origin benchmarks")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("trustworthy_outputs"))
    parser.add_argument("--months", default="all",
                        help="Comma-separated YYYY-MM; 'all' uses all measured months")
    parser.add_argument("--require-verified", action="store_true",
                        help="Optional instrument-epoch constraint")
    parser.add_argument("--min-train", type=int, default=40)
    args = parser.parse_args()
    daily = pd.read_csv(args.daily)
    months = None if args.months.lower() == "all" else tuple(
        x.strip() for x in args.months.split(",") if x.strip())
    pred, diagnostics = eligible_forecasts(
        daily, test_months=months, require_verified=args.require_verified,
        min_train=args.min_train)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "backtest_predictions.csv", index=False)
    summary = summarize(pred)
    summary.to_csv(args.out_dir / "backtest_summary.csv", index=False)
    counts = diagnostics.groupby(
        ["node", "test_month", "horizon_days", "model", "reason"]
    ).size().rename("n_days").reset_index()
    counts.to_csv(args.out_dir / "backtest_exclusions.csv", index=False)
    print(f"Evaluated {pred['node'].nunique()} / {daily['node'].nunique()} nodes;"
          f" {len(pred)} model forecasts across {len(summary)} summary groups.")
    print(f"Saved {args.out_dir / 'backtest_exclusions.csv'}")
    if pred.empty:
        print("No qualifying forecasts. Inspect backtest_exclusions.csv and"
              " station_month_coverage.csv (python -m trustworthy.coverage_report).")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
