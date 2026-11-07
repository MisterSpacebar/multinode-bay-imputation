"""Leakage-aware observed-only daily baseline backtest.

Each forecast is issued at the END of issue_date, using only completed days.
The target is the entire day at issue_date + horizon. Training targets must
finish before the issue date, and test windows are fixed before evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["hours_lt4", "odo_mean", "odo_min", "odo_range",
            "predawn_min", "temp_mean", "sal_mean", "trail7"]
HORIZONS = (1, 3, 7)
TEST_MONTHS = ("2025-09", "2026-09")


def prepare(frame: pd.DataFrame, node: str, horizon: int) -> pd.DataFrame:
    d = frame.loc[frame["node"] == node].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    if d.index.has_duplicates:
        raise ValueError(f"{node}: duplicate daily dates")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    # Historical features only. Missing days are NOT silently filled.
    d["trail7"] = d["hours_lt4"].rolling(7, min_periods=4).mean()
    d["target"] = d["hours_lt4"].shift(-horizon)
    d["target_date"] = d.index + pd.Timedelta(days=horizon)
    # Enforce same known deployment for issue and target; unknown epochs
    # are excluded by default from the trusted forecast comparison.
    d["target_epoch"] = d["deployment_id"].shift(-horizon)
    d["same_epoch"] = (
        d["source_verified"].eq(True)
        & d["source_verified"].shift(-horizon).eq(True)
        & d["deployment_id"].eq(d["target_epoch"])
    )
    return d


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float = 10.0):
    x = train[FEATURES].to_numpy(float)
    y = train["target"].to_numpy(float)
    q = test[FEATURES].to_numpy(float)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1
    x = (x - mean) / sd
    q = (q - mean) / sd
    ym = y.mean()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym))
    return np.clip(q @ w + ym, 0, 24)


def evaluate(daily: pd.DataFrame, nodes: list[str] | None = None,
             horizons=HORIZONS, test_months=TEST_MONTHS,
             require_verified: bool = True, min_train: int = 40) -> pd.DataFrame:
    rows = []
    nodes = nodes or sorted(daily["node"].unique())
    for node in nodes:
        for h in horizons:
            d = prepare(daily, node, h)
            for month in test_months:
                start = pd.Period(month).start_time
                end = pd.Period(month).end_time.normalize()
                # End-of-day forecast issue: the complete target day must
                # precede the test ISSUE date to enter model fitting.
                eligible = d["target"].notna() & d[FEATURES].notna().all(axis=1)
                if require_verified:
                    eligible &= d["same_epoch"].fillna(False)
                # Train for every test issue separately: no future target or
                # same-day target can enter the training set.
                issues = d.loc[eligible & d.index.to_series().between(
                    start - pd.Timedelta(days=h), end - pd.Timedelta(days=h)
                )]
                for issue_date, item in issues.iterrows():
                    train = d.loc[eligible & (d["target_date"] < issue_date)]
                    if len(train) < min_train:
                        continue
                    one = d.loc[[issue_date]]
                    y = float(item["target"])
                    forecasts = {
                        "persistence": float(item["hours_lt4"]),
                        "trail7": float(item["trail7"]),
                        "train_mean": float(train["target"].mean()),
                        "ridge": float(ridge_predict(train, one)[0]),
                    }
                    for name, pred in forecasts.items():
                        rows.append({
                            "node": node, "issue_date": issue_date.date().isoformat(),
                            "target_date": item["target_date"].date().isoformat(),
                            "test_month": month, "horizon_days": h,
                            "deployment_id": item["deployment_id"],
                            "model": name, "n_train": len(train),
                            "actual_hours_lt4": y, "predicted_hours_lt4": pred,
                            "absolute_error": abs(y - pred),
                            "squared_error": (y - pred) ** 2,
                        })
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    group = ["node", "test_month", "horizon_days", "model"]
    summary = predictions.groupby(group).agg(
        n=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda s: float(np.sqrt(s.mean()))),
        hypoxic_targets=("actual_hours_lt4", lambda s: int((s > 0).sum())),
    ).reset_index()
    baseline = summary[summary["model"] == "persistence"][
        ["node", "test_month", "horizon_days", "rmse"]
    ].rename(columns={"rmse": "persistence_rmse"})
    summary = summary.merge(baseline, on=group[:3], how="left")
    summary["rmse_skill_vs_persistence"] = np.where(
        summary["persistence_rmse"] > 0,
        1 - summary["rmse"] ** 2 / summary["persistence_rmse"] ** 2, np.nan
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Leakage-aware rolling-origin baseline")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("trustworthy_outputs"))
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Exploratory only: include unverified deployment epochs")
    parser.add_argument("--min-train", type=int, default=40)
    args = parser.parse_args()
    data = pd.read_csv(args.daily)
    pred = evaluate(data, require_verified=not args.allow_unverified,
                    min_train=args.min_train)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "backtest_predictions.csv", index=False)
    summary = summarize(pred)
    summary.to_csv(args.out_dir / "backtest_summary.csv", index=False)
    print(f"Predictions: {len(pred)}; station/month/horizon/model groups: {len(summary)}")
    if pred.empty:
        print("No eligible comparisons. Supply deployment epochs or use "
              "--allow-unverified for exploratory results (not validated claims).")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
