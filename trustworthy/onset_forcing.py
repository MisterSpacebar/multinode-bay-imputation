"""Optional rainfall/wind/water-level onset forecast ablations.

Requires a locally prepared provenance-aware forcing CSV accepted by
time_valid_forcing.read_forcing. Uses lag-2 values by default and compares
against *existing* trusted onset model forecasts on exactly paired issue
dates. DO/physical baselines and original FCM are not modified.

Example:
  python -m trustworthy.onset_forcing --forcing water_data/forcing_daily.csv
  python -m trustworthy.onset_forcing --forcing water_data/forcing_daily.csv \
      --allow-regional-weather

Do not interpret higher Brier skill on a small selected subset as an
operational alert. Sources with unknown publication times should not be
fabricated to make this run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.onset import (
    DO_FEATURES, PHYSICAL_FEATURES, build_windows, fit_logistic_ridge,
    summarize as risk_summary,
)
from trustworthy.time_valid_forcing import (
    read_forcing, materialize_lags,
)

MODEL_SETS = {
    "logit_rain_lag2": tuple(PHYSICAL_FEATURES) + ("rain_mm_lag2",),
    "logit_weather_lag2": tuple(PHYSICAL_FEATURES) + (
        "rain_mm_lag2", "wind_mps_lag2",
    ),
    "logit_hydro_lag2": tuple(PHYSICAL_FEATURES) + (
        "rain_mm_lag2", "wind_mps_lag2", "water_level_m_lag2",
    ),
}
REFERENCE_MODELS = ("history_rate", "logit_do", "logit_physical")
JOIN_KEYS = ("node", "issue_date", "threshold_mgL", "horizon_days")
PRED_COLUMNS = (
    "node", "issue_date", "window_end", "event_date", "threshold_mgL",
    "horizon_days", "model", "n_train", "n_train_events",
    "observed_onset", "predicted_probability", "brier_error",
    "qc_suspect_issue",
)
EXCLUSION_COLUMNS = (
    "node", "issue_date", "threshold_mgL", "horizon_days", "model",
    "reason", "n_train", "n_train_events",
)


def backtest_forcing(daily: pd.DataFrame, forcing: pd.DataFrame,
                     thresholds=(4, 2), horizons=(1, 3, 7),
                     months=None, normal_days=2, min_coverage=0.95,
                     min_train=40, min_events=3, alpha=10.0,
                     allow_regional_weather=False):
    """Return forecasts/exclusions with all fitted feature values time-valid.

    Historical outcomes are completed before each forecast issue date,
    including their entire horizon. Only complete feature cases enter each
    model's training set; models have separate training sample counts.
    """
    allowed = None if months is None else set(str(x) for x in months)
    lagged = materialize_lags(
        daily, forcing, lags=(2,),
        allow_regional_weather=allow_regional_weather,
    ).rename(columns={"date": "issue_date"})
    rows, excluded = [], []
    for node in sorted(daily["node"].unique()):
        features_by_node = lagged.loc[lagged["node"].eq(node)]
        for threshold in thresholds:
            for horizon in horizons:
                windows = build_windows(
                    daily, node, threshold, horizon,
                    normal_days=normal_days, min_coverage=min_coverage,
                )
                if windows.empty:
                    continue
                windows["issue_date"] = pd.to_datetime(windows["issue_date"])
                windows["window_end"] = pd.to_datetime(windows["window_end"])
                w = windows.merge(
                    features_by_node, on=["node", "issue_date"],
                    how="left", validate="one_to_one",
                )
                evaluated = w if allowed is None else w.loc[
                    w["issue_date"].dt.strftime("%Y-%m").isin(allowed)]
                for _, issue in evaluated.iterrows():
                    issue_date = issue["issue_date"]
                    history = w.loc[
                        w["label"].notna()
                        & w["window_end"].lt(issue_date)]
                    for name, features in MODEL_SETS.items():
                        reason = None if issue["reason"] == "evaluable" else issue["reason"]
                        if reason is None and issue[list(features)].isna().any():
                            reason = "missing_issue_features"
                        train = history.loc[
                            history[list(features)].notna().all(axis=1)]
                        n_train = len(train)
                        n_events = int(train["label"].sum()) if n_train else 0
                        if reason is None and n_train < min_train:
                            reason = "insufficient_training"
                        elif reason is None and (
                            n_events < min_events or
                            n_train - n_events < min_events
                        ):
                            reason = "insufficient_event_classes"
                        excluded.append({
                            "node": node,
                            "issue_date": issue_date.date().isoformat(),
                            "threshold_mgL": threshold,
                            "horizon_days": horizon,
                            "model": name,
                            "reason": reason or "evaluable",
                            "n_train": n_train,
                            "n_train_events": n_events,
                        })
                        if reason:
                            continue
                        probability = fit_logistic_ridge(
                            train, issue, features, alpha=alpha)
                        actual = int(issue["label"])
                        rows.append({
                            "node": node,
                            "issue_date": issue_date.date().isoformat(),
                            "window_end": issue["window_end"].date().isoformat(),
                            "event_date": issue["event_date"],
                            "threshold_mgL": threshold,
                            "horizon_days": horizon,
                            "model": name,
                            "n_train": n_train,
                            "n_train_events": n_events,
                            "observed_onset": actual,
                            "predicted_probability": probability,
                            "brier_error": (probability - actual) ** 2,
                            "qc_suspect_issue": issue["qc_suspect_issue"],
                        })
    return (pd.DataFrame(rows, columns=PRED_COLUMNS),
            pd.DataFrame(excluded, columns=EXCLUSION_COLUMNS))


def paired_comparisons(forcing_predictions: pd.DataFrame,
                       reference_predictions: pd.DataFrame) -> pd.DataFrame:
    """On exactly matching station/time/horizon, compare Brier to old models."""
    if forcing_predictions.empty or reference_predictions.empty:
        return pd.DataFrame()
    if forcing_predictions.duplicated(list(JOIN_KEYS) + ["model"]).any():
        raise ValueError("Duplicate forcing forecast")
    if reference_predictions.duplicated(list(JOIN_KEYS) + ["model"]).any():
        raise ValueError("Duplicate baseline forecast")
    reference = reference_predictions.loc[
        reference_predictions["model"].isin(REFERENCE_MODELS),
        list(JOIN_KEYS) + ["model", "observed_onset", "predicted_probability"]
    ]
    left = forcing_predictions[list(JOIN_KEYS) + [
        "model", "observed_onset", "predicted_probability"
    ]]
    joined = left.merge(reference, on=list(JOIN_KEYS),
                        suffixes=("_forcing", "_ref"),
                        how="inner", validate="many_to_many")
    if (joined["observed_onset_forcing"] != joined["observed_onset_ref"]).any():
        raise ValueError("Reference outcomes do not match forcing backtest")
    if joined.empty:
        return pd.DataFrame()
    joined["forcing_sq"] = (
        joined["predicted_probability_forcing"]
        - joined["observed_onset_forcing"]) ** 2
    joined["baseline_sq"] = (
        joined["predicted_probability_ref"]
        - joined["observed_onset_forcing"]) ** 2
    rows = []
    for key, group in joined.groupby([
        "threshold_mgL", "horizon_days", "model_forcing", "model_ref",
    ]):
        a, b = float(group["forcing_sq"].mean()), float(group["baseline_sq"].mean())
        rows.append({
            "threshold_mgL": key[0], "horizon_days": key[1],
            "candidate": key[2], "reference": key[3],
            "n_paired": len(group),
            "n_stations": group["node"].nunique(),
            "n_positive_windows": int(group["observed_onset_forcing"].sum()),
            "candidate_brier": a, "reference_brier": b,
            "brier_skill_vs_reference": 1.0 - a / b if b > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="Optional issue-time-valid environmental onset models")
    p.add_argument("--daily", type=Path, default=Path(
        "trustworthy_outputs/daily_targets.csv"))
    p.add_argument("--forcing", type=Path, required=True)
    p.add_argument("--baselines", type=Path, default=Path(
        "trustworthy_outputs/onset/onset_predictions.csv"))
    p.add_argument("--out-dir", type=Path, default=Path(
        "trustworthy_outputs/onset/forcing"))
    p.add_argument("--months", default="all",
                   help="All or comma-separated issue months")
    p.add_argument("--allow-regional-weather", action="store_true")
    p.add_argument("--min-train", type=int, default=40)
    p.add_argument("--min-events", type=int, default=3)
    args = p.parse_args()
    daily = pd.read_csv(args.daily)
    observations = read_forcing(args.forcing)
    months = None if args.months.lower() == "all" else [
        x.strip() for x in args.months.split(",") if x.strip()]
    preds, exclusions = backtest_forcing(
        daily, observations, months=months,
        min_train=args.min_train, min_events=args.min_events,
        allow_regional_weather=args.allow_regional_weather)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out_dir / "predictions.csv", index=False)
    exclusions.groupby([
        "node", "threshold_mgL", "horizon_days", "model", "reason"
    ]).size().rename("n_issues").reset_index().to_csv(
        args.out_dir / "exclusions.csv", index=False)
    risk_summary(preds).to_csv(args.out_dir / "summary.csv", index=False)
    if args.baselines.exists() and not preds.empty:
        refs = pd.read_csv(args.baselines)
        paired_comparisons(preds, refs).to_csv(
            args.out_dir / "paired_baselines.csv", index=False)
    else:
        print("No baseline forecasts or forcing forecasts; paired report skipped")
    print(f"Environmental forcing forecasts: {len(preds)}")
    print(f"Full diagnostics: {args.out_dir}")


if __name__ == "__main__":
    main()
