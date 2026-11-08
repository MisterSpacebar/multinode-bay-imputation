"""Exploratory paired Brier uncertainty with station/time-block resampling.

Preserves ALL forecasts from the same station and calendar block together.
This accounts for dependence within each sampled block, including overlapping
onset windows. Correlation across blocks/stations remains and estimates are
NOT confirmatory confidence intervals.

Usage:
  python -m trustworthy.onset_uncertainty
  python -m trustworthy.onset_uncertainty --bootstrap 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("node", "issue_date", "threshold_mgL", "horizon_days")
COMPARISONS = (
    ("logit_do", "history_rate"),
    ("logit_physical", "history_rate"),
    ("logit_physical", "logit_do"),
)
REPORT_COLUMNS = (
    "threshold_mgL", "horizon_days", "candidate", "reference",
    "n_forecast_windows", "n_positive_windows", "n_distinct_onsets",
    "n_stations", "n_station_month_blocks", "block_scheme",
    "n_bootstrap", "candidate_brier", "reference_brier",
    "delta_brier", "delta_brier_ci_low_approx",
    "delta_brier_ci_high_approx", "brier_skill",
    "brier_skill_ci_low_approx", "brier_skill_ci_high_approx",
    "interval_status",
)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS) | {
        "model", "observed_onset", "predicted_probability", "event_date"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing onset prediction columns: {sorted(missing)}")
    d = predictions.copy()
    d["issue_date"] = pd.to_datetime(d["issue_date"], errors="raise")
    if d["issue_date"].isna().any():
        raise ValueError("Invalid issue dates")
    if d.duplicated(list(KEYS) + ["model"]).any():
        raise ValueError("Duplicated station/issue/threshold/horizon/model")
    for field in ("observed_onset", "predicted_probability"):
        d[field] = pd.to_numeric(d[field], errors="raise")
        if not np.isfinite(d[field]).all():
            raise ValueError(f"{field} contains missing/nonfinite values")
    if not d["observed_onset"].isin([0, 1]).all():
        raise ValueError("Onset labels must be 0 or 1")
    if not d["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be in [0, 1]")
    if d.loc[d["observed_onset"].eq(1), "event_date"].isna().any():
        raise ValueError("Positive windows require a dated observed onset")
    return d


def paired_cases(d: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    columns = list(KEYS) + ["observed_onset", "predicted_probability", "event_date"]
    left = d.loc[d["model"].eq(a), columns].rename(columns={
        "observed_onset": "actual_a",
        "predicted_probability": "candidate_probability",
        "event_date": "event_a",
    })
    right = d.loc[d["model"].eq(b), columns].rename(columns={
        "observed_onset": "actual_b",
        "predicted_probability": "reference_probability",
        "event_date": "event_b",
    })
    both = left.merge(right, on=list(KEYS), how="inner",
                      validate="one_to_one")
    if not both.empty:
        if not (both["actual_a"] == both["actual_b"]).all():
            raise ValueError("Inconsistent observed onset labels")
        positives = both["actual_a"].eq(1)
        if not (both.loc[positives, "event_a"].astype(str).to_numpy()
                == both.loc[positives, "event_b"].astype(str).to_numpy()).all():
            raise ValueError("Inconsistent observed event onset dates")
        both["candidate_sq"] = (
            both["candidate_probability"] - both["actual_a"]) ** 2
        both["reference_sq"] = (
            both["reference_probability"] - both["actual_a"]) ** 2
        both["month"] = both["issue_date"].dt.to_period("M").astype(str)
    return both


def _blocks(cases: pd.DataFrame, scheme: str) -> dict[str, np.ndarray]:
    if scheme == "month":
        block = cases["month"]
    elif scheme in ("14day", "28day"):
        days = int(scheme.removesuffix("day"))
        # Fixed nonoverlapping calendar-time blocks, not N available rows.
        block = ((cases["issue_date"] - pd.Timestamp("1970-01-01"))
                 .dt.days // days).astype(str)
    else:
        raise ValueError(f"Unknown block scheme: {scheme}")
    working = cases.assign(block=block.to_numpy())
    groups = {}
    for (node, block_id), g in working.groupby(
            ["node", "block"], sort=True):
        groups.setdefault(str(node), []).append([
            float(g["candidate_sq"].sum()),
            float(g["reference_sq"].sum()),
            len(g),
        ])
    return {node: np.asarray(values, dtype=float)
            for node, values in groups.items()}


def bootstrap_brier(cases: pd.DataFrame, scheme: str,
                    n_bootstrap: int = 2000, seed: int = 2026):
    """Station-stratified block bootstrap of candidate-reference Brier skill.

    Sampling station blocks (not individual forecasts) preserves within-block
    dependence, but neither cross-station nor across-block dependence.
    """
    if n_bootstrap < 100:
        raise ValueError("At least 100 draws needed for percentile intervals")
    groups = _blocks(cases, scheme)
    rng = np.random.default_rng(seed)
    delta, skill = [], []
    for _ in range(n_bootstrap):
        totals = np.zeros(3, dtype=float)
        for block_values in groups.values():
            draw = rng.integers(0, len(block_values), size=len(block_values))
            totals += block_values[draw].sum(axis=0)
        if totals[2] <= 0 or totals[1] <= 0:
            continue
        mse_a = totals[0] / totals[2]
        mse_b = totals[1] / totals[2]
        delta.append(mse_a - mse_b)
        skill.append(1.0 - mse_a / mse_b)
    if len(delta) < n_bootstrap * 0.95:
        return (np.nan,) * 4, "degenerate_resamples"
    dlo, dhi = np.quantile(delta, [0.025, 0.975])
    slo, shi = np.quantile(skill, [0.025, 0.975])
    return (float(dlo), float(dhi), float(slo), float(shi)), "approximate"


def summarize_uncertainty(predictions: pd.DataFrame, n_bootstrap=2000,
                          seed=2026, schemes=("month", "14day", "28day"),
                          min_station_blocks=8) -> pd.DataFrame:
    d = validate_predictions(predictions)
    rows = []
    for (th, h), grp in d.groupby(
            ["threshold_mgL", "horizon_days"], sort=True):
        for a, b in COMPARISONS:
            paired = paired_cases(grp, a, b)
            if paired.empty:
                continue
            aa, bb = (float(paired["candidate_sq"].mean()),
                      float(paired["reference_sq"].mean()))
            positives = paired["actual_a"].eq(1)
            onsets = (paired.loc[positives, ["node", "event_a"]]
                      .drop_duplicates().shape[0])
            n_blocks = paired.groupby(["node", "month"]).ngroups
            for scheme in schemes:
                block_count = sum(map(len, _blocks(paired, scheme).values()))
                if block_count < min_station_blocks:
                    bounds, status = (np.nan,) * 4, "too_few_time_blocks"
                elif bb <= 0:
                    bounds, status = (np.nan,) * 4, "zero_reference_brier"
                else:
                    bounds, status = bootstrap_brier(
                        paired, scheme, n_bootstrap, seed +
                        int(th) * 1000 + int(h) * 10)
                rows.append({
                    "threshold_mgL": int(th), "horizon_days": int(h),
                    "candidate": a, "reference": b,
                    "n_forecast_windows": len(paired),
                    "n_positive_windows": int(positives.sum()),
                    "n_distinct_onsets": int(onsets),
                    "n_stations": paired["node"].nunique(),
                    "n_station_month_blocks": n_blocks,
                    "block_scheme": scheme,
                    "n_bootstrap": n_bootstrap if status == "approximate" else 0,
                    "candidate_brier": aa, "reference_brier": bb,
                    "delta_brier": aa - bb,
                    "delta_brier_ci_low_approx": bounds[0],
                    "delta_brier_ci_high_approx": bounds[1],
                    "brier_skill": 1 - aa / bb if bb > 0 else np.nan,
                    "brier_skill_ci_low_approx": bounds[2],
                    "brier_skill_ci_high_approx": bounds[3],
                    "interval_status": status,
                })
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Approximate paired Brier skill uncertainty")
    parser.add_argument("--predictions", type=Path, default=Path(
        "trustworthy_outputs/onset/onset_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path(
        "trustworthy_outputs/onset/onset_uncertainty.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = summarize_uncertainty(
        pd.read_csv(args.predictions), n_bootstrap=args.bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result)} paired uncertainty rows to {args.out}")
    print(result.loc[
        result["threshold_mgL"].eq(2)
        & result["block_scheme"].eq("month"),
        ["horizon_days", "candidate", "reference", "n_forecast_windows",
         "n_distinct_onsets", "brier_skill", "brier_skill_ci_low_approx",
         "brier_skill_ci_high_approx", "interval_status"],
    ].to_string(index=False))


if __name__ == "__main__":
    main()
