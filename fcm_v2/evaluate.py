"""Backtest station-aware physical FCMs on the trustworthy observed-only targets.

Run:
  python -m fcm_v2.evaluate
  python -m fcm_v2.evaluate --months 2025-09,2026-09
  python -m fcm_v2.evaluate --activation linear

No deployment CSV, imputed GNN fields, retrospective FCM forcing, future
observation, or fitted normalization from the test period is used. Station
models are independently fitted per issue date and horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fcm_v2.model import (
    DO_CONCEPTS, PHYSICAL_CONCEPTS, edge_rows,
    fit_map, prepare_targets, sensitivity_rows,
)
from trustworthy.backtest import prepare, summarize as summarize_baselines

PREDICTION_COLUMNS = (
    "node", "issue_date", "target_date", "test_month", "horizon_days",
    "model", "activation", "concept_set", "n_train",
    "actual_hours_lt4", "predicted_hours_lt4", "persistence_prediction",
    "absolute_error", "squared_error",
)
EXCLUSION_COLUMNS = (
    "node", "issue_date", "target_date", "test_month",
    "horizon_days", "model", "reason", "n_train",
)
SUMMARY_COLUMNS = (
    "node", "test_month", "horizon_days", "model", "n_test",
    "n_hypoxic_target_days", "mae_hours", "rmse_hours",
    "hypoxic_day_mae_hours", "paired_persistence_mae",
    "paired_persistence_rmse", "rmse_skill_vs_persistence",
)


def model_configs(activation="both"):
    if activation not in ("both", "linear", "sigmoid"):
        raise ValueError("activation must be both, linear, or sigmoid")
    # DO-only map isolates oxygen autocorrelation from measured physical
    # relationships. All variants predict the same observed hypoxia target.
    options = []
    for act in ("linear", "sigmoid"):
        if activation in ("both", act):
            options.append((f"fcm_{act}_do", DO_CONCEPTS, act))
            options.append((f"fcm_{act}_physical", PHYSICAL_CONCEPTS, act))
    return options


def evaluate(daily: pd.DataFrame, horizons=(1, 3, 7),
             months=None, activation="both", min_train=40, alpha=10.0,
             include_edges=True):
    """Return predictions, full exclusion diagnostics, learned edges, sensitivities.

    Every target date in a station's first-to-last date range is considered.
    For each issue, its train set contains only observed target pairs whose
    target_date precedes the issue_date. Test dates are never used for fitting.
    """
    if not {"node", "date", "hours_lt4"}.issubset(daily.columns):
        raise ValueError("daily CSV must have node,date,hours_lt4 columns")
    if min_train < 2:
        raise ValueError("min_train must be >=2")
    if months is None:
        months = sorted(pd.to_datetime(daily["date"])
                        .dt.to_period("M").astype(str).unique())
    configs = model_configs(activation)
    rows, exclusions, weights, scenarios = [], [], [], []
    for node in sorted(daily["node"].unique()):
        for horizon in horizons:
            # Reuse the identical calendar-day construction as trustworthy.
            frame = prepare(daily, node, horizon)
            if frame.empty:
                continue
            # Never interpolate missing values; only complete case fitting.
            # Existing raw QC is already observed-only and coverage gated.
            frame = prepare_targets(frame, PHYSICAL_CONCEPTS, horizon)
            for month in months:
                period = pd.Period(str(month), "M")
                target_days = pd.date_range(
                    max(frame.index.min(), period.start_time),
                    min(frame.index.max(), period.end_time.normalize()),
                    freq="D",
                )
                for target_date in target_days:
                    issued_at = target_date - pd.Timedelta(days=horizon)
                    target = float(frame.at[target_date, "hours_lt4"])
                    for model_name, concepts, act in configs:
                        count = 0
                        if not np.isfinite(target):
                            reason = "missing_target"
                        elif issued_at not in frame.index:
                            reason = "no_issue_history"
                        elif not np.isfinite(frame.at[issued_at, "hours_lt4"]):
                            reason = "missing_issue_do"
                        elif not frame.loc[issued_at, list(concepts)].notna().all():
                            reason = "missing_model_features"
                        else:
                            reason = None
                        cmap = None
                        if reason is None:
                            # A complete issue-day state is necessary, but
                            # missing physical targets at other dates do not
                            # imply missing hypoxia labels on these dates.
                            cmap = fit_map(
                                frame, tuple(concepts), horizon, issued_at,
                                activation=act, min_train=min_train, alpha=alpha,
                            )
                            if cmap is None:
                                reason = "insufficient_training"
                            else:
                                count = int(cmap.n_target_pairs[0])
                        exclusions.append({
                            "node": node, "issue_date": issued_at.date().isoformat(),
                            "target_date": target_date.date().isoformat(),
                            "test_month": str(month), "horizon_days": horizon,
                            "model": model_name, "reason": reason or "evaluable",
                            "n_train": count,
                        })
                        if reason is not None:
                            continue
                        current = frame.loc[issued_at, list(concepts)].to_numpy(
                            dtype=float)
                        prediction = float(cmap.predict(current)[0])
                        persistence = float(frame.at[issued_at, "hours_lt4"])
                        record = {
                            "node": node, "issue_date": issued_at.date().isoformat(),
                            "target_date": target_date.date().isoformat(),
                            "test_month": str(month), "horizon_days": horizon,
                            "model": model_name, "activation": act,
                            "concept_set": ("physical" if len(concepts) > 2 else "do"),
                            "n_train": count, "actual_hours_lt4": target,
                            "predicted_hours_lt4": prediction,
                            "persistence_prediction": persistence,
                            "absolute_error": abs(target - prediction),
                            "squared_error": (target - prediction) ** 2,
                        }
                        rows.append(record)
                        if include_edges:
                            weights.extend(edge_rows(cmap, node, issued_at))
                            scenarios.extend(sensitivity_rows(
                                cmap, node, issued_at, current))
    return (
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS),
        pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS),
        pd.DataFrame(weights, columns=[
            "node", "asof_date", "horizon_days", "activation",
            "concept_set", "source", "target", "weight", "intercept",
            "n_train_target", "weight_units", "interpretation",
        ]),
        pd.DataFrame(scenarios, columns=[
            "node", "asof_date", "horizon_days", "activation",
            "perturbed_concept", "requested_delta", "actual_delta",
            "base_predicted_hours_lt4", "altered_predicted_hours_lt4",
            "predicted_difference_hours", "interpretation",
        ]),
    )


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return summarize_baselines(predictions)


def compare_on_common_cases(fcm_preds: pd.DataFrame,
                            baseline_preds: pd.DataFrame) -> pd.DataFrame:
    """Restrict all models to dates where every candidate is available.

    This is intentionally a stricter *common-case* report; use paired
    persistence skill in the regular summary for broader model coverage.
    No row is compared across stations or horizons.
    """
    keys = ["node", "issue_date", "target_date", "horizon_days"]
    if fcm_preds.empty or baseline_preds.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    candidates = pd.concat([
        fcm_preds[keys + ["model", "actual_hours_lt4",
                          "predicted_hours_lt4", "persistence_prediction"]],
        baseline_preds[keys + ["model", "actual_hours_lt4",
                               "predicted_hours_lt4", "persistence_prediction"]],
    ], ignore_index=True)
    if candidates.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate prediction issue/target/model entries")
    expected = int(candidates["model"].nunique())
    valid_cases = (candidates.groupby(keys)["model"].nunique()
                   .loc[lambda x: x == expected].reset_index()[keys])
    selected = candidates.merge(valid_cases, on=keys, how="inner")
    if selected.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    selected["test_month"] = pd.to_datetime(
        selected["target_date"]).dt.to_period("M").astype(str)
    return summarize(selected)


def main():
    parser = argparse.ArgumentParser(description="Observed-only physical FCM v2")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--baselines", type=Path,
                        default=Path("trustworthy_outputs/backtest_predictions.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("fcm_v2_outputs"))
    parser.add_argument("--months", default="all",
                        help="all or comma-separated YYYY-MM")
    parser.add_argument("--activation", choices=["both", "linear", "sigmoid"],
                        default="both")
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--no-edges", action="store_true",
                        help="Skip writing rolling edge/sensitivity history")
    args = parser.parse_args()
    data = pd.read_csv(args.daily)
    months = None if args.months.lower() == "all" else [
        p.strip() for p in args.months.split(",") if p.strip()]
    p, d, edges, s = evaluate(
        data, months=months, min_train=args.min_train,
        alpha=args.alpha, activation=args.activation,
        include_edges=not args.no_edges,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p.to_csv(args.out_dir / "predictions.csv", index=False)
    d.groupby(["node", "test_month", "horizon_days", "model", "reason"]
              ).size().rename("n_days").reset_index().to_csv(
        args.out_dir / "exclusions.csv", index=False)
    summary = summarize(p)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    edges.to_csv(args.out_dir / "learned_edges.csv", index=False)
    s.to_csv(args.out_dir / "scenario_sensitivity.csv", index=False)
    if args.baselines.exists():
        b = pd.read_csv(args.baselines)
        common = compare_on_common_cases(p, b)
        common.to_csv(args.out_dir / "all_models_common_cases.csv", index=False)
        print(f"All-model common-case groups: {len(common)}")
    else:
        print(f"Baseline predictions missing: {args.baselines}")
    print(f"FCM predictions: {len(p)} over {p['node'].nunique()} stations")
    print(f"Saved reports to {args.out_dir}")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No eligible station-days. Inspect exclusions.csv.")


if __name__ == "__main__":
    main()
