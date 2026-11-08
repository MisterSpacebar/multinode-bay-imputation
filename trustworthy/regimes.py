"""Coverage-gated observed-only hypoxia regime labels and forecast score tables.

A day is hypoxic if ANY observed five-minute sample was < threshold, as
represented by hours_lt4 / hours_lt2 from trustworthy.quality. Regimes are
assigned by comparing the current day with its immediately preceding CALENDAR
day. Unknown or insufficient-coverage days never become normal by default.

This is retrospective outcome stratification, not a feature available when
making future predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = {4: "hours_lt4", 2: "hours_lt2"}
REGIMES = ("normal", "onset", "persistent_hypoxia", "recovery", "unknown")
KEYS = ("node", "issue_date", "target_date", "horizon_days", "model")
REGIME_COLUMNS = (
    "node", "date", "threshold_mgL", "coverage", "hours_below",
    "current_hypoxic", "prior_hypoxic", "regime", "valid_transition",
    "qc_suspect", "eligible",
)


def station_calendar(daily: pd.DataFrame, node: str) -> pd.DataFrame:
    """Complete station-local calendar; no future/past filling of observations."""
    needed = {"node", "date", "coverage", "eligible"}
    missing = needed - set(daily)
    if missing:
        raise ValueError(f"Missing daily target columns: {sorted(missing)}")
    d = daily.loc[daily["node"] == node].copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    if d["date"].isna().any() or d["date"].dt.tz is not None:
        raise ValueError("Daily dates must be valid timezone-naive local dates")
    d = d.set_index("date").sort_index()
    if d.index.has_duplicates:
        raise ValueError(f"Duplicate dates for station {node}")
    # Do not extend before the first or after the last observed station date.
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d["eligible"] = (d["eligible"].astype("string").str.lower()
                     .eq("true").fillna(False))
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce")
    for col in THRESHOLDS.values():
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def valid_hours(d: pd.DataFrame, threshold: int,
                min_coverage: float = 0.95) -> pd.Series:
    if threshold not in THRESHOLDS:
        raise ValueError("threshold must be 2 or 4 mg/L")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0,1]")
    hours = d[THRESHOLDS[threshold]]
    return (d["eligible"].eq(True) & d["coverage"].ge(min_coverage)
            & hours.notna() & np.isfinite(hours)
            & hours.ge(0) & hours.le(24))


def regime_days(daily: pd.DataFrame, thresholds=(4, 2),
                min_coverage=0.95) -> pd.DataFrame:
    """Return one row for every station/date/threshold, including unknown."""
    rows = []
    for node in sorted(daily["node"].unique()):
        d = station_calendar(daily, node)
        if d.empty:
            continue
        for threshold in thresholds:
            valid = valid_hours(d, threshold, min_coverage)
            active = d[THRESHOLDS[threshold]].gt(0)
            prev_valid = valid.shift(1, fill_value=False)
            previous = active.shift(1, fill_value=False)
            transition = valid & prev_valid
            labels = np.full(len(d), "unknown", dtype=object)
            labels[transition & ~previous & ~active] = "normal"
            labels[transition & ~previous & active] = "onset"
            labels[transition & previous & active] = "persistent_hypoxia"
            labels[transition & previous & ~active] = "recovery"
            suspect = pd.Series(False, index=d.index)
            for col in ("n_flatline", "n_spike_suspect", "n_zero_review"):
                if col in d:
                    suspect |= pd.to_numeric(
                        d[col], errors="coerce").fillna(0).gt(0)
            rows.append(pd.DataFrame({
                "node": node, "date": d.index.strftime("%Y-%m-%d"),
                "threshold_mgL": threshold,
                "coverage": d["coverage"].to_numpy(float),
                "hours_below": d[THRESHOLDS[threshold]].to_numpy(float),
                "current_hypoxic": active.where(valid, pd.NA).to_numpy(),
                "prior_hypoxic": previous.where(prev_valid, pd.NA).to_numpy(),
                "regime": labels,
                "valid_transition": transition.to_numpy(bool),
                "qc_suspect": suspect.to_numpy(bool),
                "eligible": valid.to_numpy(bool),
            }))
    return (pd.concat(rows, ignore_index=True)[list(REGIME_COLUMNS)]
            if rows else pd.DataFrame(columns=REGIME_COLUMNS))


def regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    """Station-month event and unknown counts; not independent event incidence."""
    if regimes.empty:
        return pd.DataFrame()
    d = regimes.copy()
    d["month"] = d["date"].str.slice(0, 7)
    counts = d.groupby(
        ["node", "month", "threshold_mgL", "regime"],
        dropna=False,
    ).size().rename("n_days").reset_index()
    return counts


def stratified_forecast_scores(regimes: pd.DataFrame,
                               predictions: pd.DataFrame,
                               threshold: int = 4,
                               by_station: bool = False) -> pd.DataFrame:
    """Target-day regime conditioned evaluation of continuous hours<4 forecasts.

    Regime is a *RETROSPECTIVE* target-day classification, never usable at
    issue time. For forecasts of hours_lt4, use threshold 4 only.
    """
    if threshold != 4:
        raise ValueError("Current continuous forecasts target hours_lt4; use threshold 4")
    cols = list(KEYS) + [
        "actual_hours_lt4", "predicted_hours_lt4", "persistence_prediction"]
    if not set(cols).issubset(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(set(cols)-set(predictions))}")
    if predictions.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate issue/target/model predictions")
    subset = regimes.loc[
        regimes["threshold_mgL"].eq(threshold),
        ["node", "date", "regime", "qc_suspect"],
    ].rename(columns={"date": "target_date"})
    merged = predictions[cols].merge(
        subset, on=["node", "target_date"], how="left", validate="many_to_one"
    )
    merged["regime"] = merged["regime"].fillna("unknown")
    merged["sq_error"] = (
        merged["actual_hours_lt4"] - merged["predicted_hours_lt4"]) ** 2
    merged["base_sq_error"] = (
        merged["actual_hours_lt4"] - merged["persistence_prediction"]) ** 2
    merged["abs_error"] = np.abs(
        merged["actual_hours_lt4"] - merged["predicted_hours_lt4"])
    group = (["node"] if by_station else []) + [
        "horizon_days", "model", "regime"]
    records = []
    for key, g in merged.groupby(group, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = dict(zip(group, key))
        mse = float(g["sq_error"].mean())
        base_mse = float(g["base_sq_error"].mean())
        values.update({
            "n_cases": len(g),
            "n_stations": g["node"].nunique(),
            "n_suspect_target_days": int(g["qc_suspect"].fillna(False).sum()),
            "n_positive_hours_targets": int(g["actual_hours_lt4"].gt(0).sum()),
            "mae_hours": float(g["abs_error"].mean()),
            "rmse_hours": float(np.sqrt(mse)),
            "paired_persistence_rmse_hours": float(np.sqrt(base_mse)),
            "rmse_skill_vs_persistence": (
                float(1.0 - mse / base_mse) if base_mse > 0 else np.nan),
        })
        records.append(values)
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description="Observed-only hypoxia day regimes")
    p.add_argument("--daily", type=Path,
                   default=Path("trustworthy_outputs/daily_targets.csv"))
    p.add_argument("--baseline-predictions", type=Path,
                   default=Path("trustworthy_outputs/backtest_predictions.csv"))
    p.add_argument("--fcm-predictions", type=Path,
                   default=Path("fcm_v2_outputs/predictions.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("trustworthy_outputs/onset"))
    p.add_argument("--min-coverage", type=float, default=0.95)
    args = p.parse_args()
    d = pd.read_csv(args.daily)
    r = regime_days(d, min_coverage=args.min_coverage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out_dir / "regime_days.csv", index=False)
    regime_summary(r).to_csv(args.out_dir / "regime_summary.csv", index=False)
    forecast_files = [x for x in (
        args.baseline_predictions, args.fcm_predictions) if x.exists()]
    if forecast_files:
        all_preds = pd.concat(
            [pd.read_csv(path) for path in forecast_files], ignore_index=True)
        stratified_forecast_scores(r, all_preds).to_csv(
            args.out_dir / "regime_forecast_pooled.csv", index=False)
        stratified_forecast_scores(r, all_preds, by_station=True).to_csv(
            args.out_dir / "regime_forecast_station.csv", index=False)
    else:
        print("No forecast CSVs: regime labels created; score tables skipped")
    print(f"Station-day regime rows: {len(r)}; stations: {r.node.nunique()}")
    print(f"Saved files to {args.out_dir}")


if __name__ == "__main__":
    main()
