"""Station/month observed-only coverage diagnostics for the hypoxia workflow.

Usage:
  python -m trustworthy.coverage_report
  python -m trustworthy.coverage_report --daily trustworthy_outputs/daily_targets.csv

Missing months between a station's first/last observation are included.
Counts are on LOCAL calendar days as recorded by trustworthy.quality.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "node", "month", "calendar_days", "recorded_days", "eligible_days",
    "eligible_fraction", "days_with_observed_hypoxia", "days_with_severe_hypoxia",
    "do_available_days", "predawn_available_days", "temperature_available_days",
    "salinity_available_days", "all_full_model_features_days",
    "missing_do_days", "low_coverage_days", "high_suspect_days",
    "mean_do_coverage", "observed_hypoxia_hours", "observed_severe_hours",
]


def coverage_report(daily: pd.DataFrame) -> pd.DataFrame:
    """Return explicit station-month accounting, without filling targets."""
    if daily.empty:
        return pd.DataFrame(columns=COLUMNS)
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    rows = []
    for node, group in data.groupby("node", sort=True):
        group = group.sort_values("date").set_index("date")
        if group.index.has_duplicates:
            raise ValueError(f"Duplicate dates for node {node}")
        complete = group.reindex(pd.date_range(
            group.index.min(), group.index.max(), freq="D"))
        complete["recorded"] = complete.index.isin(group.index)
        for month, sub in complete.groupby(complete.index.to_period("M")):
            def available(name):
                if name not in sub.columns:
                    return pd.Series(False, index=sub.index)
                return pd.to_numeric(sub[name], errors="coerce").notna()

            eligibility = sub.get(
                "eligible", pd.Series(False, index=sub.index)
            ).astype("string").str.lower().eq("true").fillna(False)
            do = available("hours_lt4")
            predawn = available("predawn_min")
            temp = available("temp_mean")
            sal = available("sal_mean")
            full = (do & predawn & temp & sal
                    & available("odo_mean") & available("odo_min")
                    & available("odo_range"))
            cov = pd.to_numeric(sub.get(
                "coverage", pd.Series(np.nan, index=sub.index)),
                errors="coerce")
            flags = pd.to_numeric(sub.get(
                "n_flatline", pd.Series(0, index=sub.index)),
                errors="coerce").fillna(0) + pd.to_numeric(sub.get(
                "n_spike_suspect", pd.Series(0, index=sub.index)),
                errors="coerce").fillna(0)
            lt4 = pd.to_numeric(sub.get(
                "hours_lt4", pd.Series(np.nan, index=sub.index)),
                errors="coerce")
            lt2 = pd.to_numeric(sub.get(
                "hours_lt2", pd.Series(np.nan, index=sub.index)),
                errors="coerce")
            rows.append({
                "node": node, "month": str(month),
                "calendar_days": len(sub),
                "recorded_days": int(sub["recorded"].sum()),
                "eligible_days": int(eligibility.sum()),
                "eligible_fraction": float(eligibility.mean()),
                "days_with_observed_hypoxia": int((lt4 > 0).sum()),
                "days_with_severe_hypoxia": int((lt2 > 0).sum()),
                "do_available_days": int(do.sum()),
                "predawn_available_days": int(predawn.sum()),
                "temperature_available_days": int(temp.sum()),
                "salinity_available_days": int(sal.sum()),
                "all_full_model_features_days": int(full.sum()),
                "missing_do_days": int((~do).sum()),
                "low_coverage_days": int((sub["recorded"] & ~eligibility).sum()),
                "high_suspect_days": int((flags > 0).sum()),
                "mean_do_coverage": float(cov.mean()) if cov.notna().any() else np.nan,
                "observed_hypoxia_hours": float(lt4.sum(min_count=1)),
                "observed_severe_hours": float(lt2.sum(min_count=1)),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Observed station-month coverage")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("trustworthy_outputs/station_month_coverage.csv"))
    args = parser.parse_args()
    daily = pd.read_csv(args.daily)
    result = coverage_report(daily)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result)} station-months over "
          f"{result['node'].nunique()} stations to {args.out}")
    print("Eligible days by station:")
    print(result.groupby("node")["eligible_days"].sum().to_string())


if __name__ == "__main__":
    main()
