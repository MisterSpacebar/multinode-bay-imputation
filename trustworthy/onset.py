"""Forecast *new-onset* hypoxia risk, distinct from persistent hypoxia.

A forecast is issued at the end of a fully observed, normal local day,
after a configurable number of consecutive eligible normal days (default 2).
Its binary outcome is whether the station first enters hypoxia on ANY of
the following 1, 3 or 7 complete local calendar days.

Both positive and negative labels require every future day to pass the
minimum observed-DO coverage gate. No filling of unknown future days.
An at-risk origin must have observed normal DO at its own threshold;
for <2 mg/L a currently <4 but >=2 day is still eligible.

Only *historically completed* outcome windows (window_end < issue date)
are used to fit a model or estimate the historical event-rate baseline.
Logistic coefficients and scaling are trained on those observations only.
This is a retrospective risk benchmark, not a validated clinical/operational
alert. Multiple overlapping horizon labels from a single episode are not
statistically independent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.regimes import THRESHOLDS, station_calendar, valid_hours

HORIZONS = (1, 3, 7)
DO_FEATURES = ("odo_mean", "odo_min", "odo_range", "hours_lt4", "hours_lt2")
PHYSICAL_FEATURES = DO_FEATURES + ("temp_mean", "sal_mean")
MODEL_FEATURES = {
    "history_rate": (),
    "logit_do": DO_FEATURES,
    "logit_physical": PHYSICAL_FEATURES,
}
PRED_COLUMNS = (
    "node", "issue_date", "window_end", "event_date", "threshold_mgL",
    "horizon_days", "model", "n_train", "n_train_events",
    "observed_onset", "predicted_probability", "brier_error",
    "qc_suspect_issue",
)
WINDOW_COLUMNS = (
    "node", "issue_date", "window_end", "threshold_mgL", "horizon_days",
    "label", "event_date", "reason", "qc_suspect_issue",
)
EXCLUSION_COLUMNS = (
    "node", "issue_date", "threshold_mgL", "horizon_days",
    "model", "reason", "n_train", "n_train_events",
)


def build_windows(daily: pd.DataFrame, node: str, threshold: int,
                  horizon: int, normal_days: int = 2,
                  min_coverage: float = 0.95) -> pd.DataFrame:
    """Create one row for each potential end-of-day issue.

    Full forecast windows are labelled only when every day has sufficient
    observed DO coverage. No event can be declared on an unobserved date.
    """
    if normal_days < 1 or horizon < 1:
        raise ValueError("normal_days and horizon must be >= 1")
    d = station_calendar(daily, node)
    if d.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS + DO_FEATURES + (
            "temp_mean", "sal_mean"))
    valid = valid_hours(d, threshold, min_coverage)
    below = d[THRESHOLDS[threshold]].gt(0)
    for col in PHYSICAL_FEATURES:
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    suspect = pd.Series(False, index=d.index)
    for col in ("n_flatline", "n_spike_suspect", "n_zero_review"):
        if col in d:
            suspect |= pd.to_numeric(
                d[col], errors="coerce").fillna(0).gt(0)

    rows = []
    for pos, issue in enumerate(d.index):
        reason, label, event_date = None, np.nan, ""
        if pos < normal_days - 1:
            reason = "no_baseline_history"
        elif not valid.iloc[pos - normal_days + 1:pos + 1].all():
            reason = "baseline_low_coverage"
        elif below.iloc[pos - normal_days + 1:pos + 1].any():
            reason = "already_below_threshold"
        elif pos + horizon >= len(d):
            reason = "incomplete_future_window"
        elif not valid.iloc[pos + 1:pos + horizon + 1].all():
            reason = "future_low_coverage"
        else:
            future = below.iloc[pos + 1:pos + horizon + 1]
            label = int(future.any())
            if label:
                event_date = future.index[np.flatnonzero(future.to_numpy())[0]].date().isoformat()
            reason = "evaluable"
        row = {
            "node": node, "issue_date": issue.date().isoformat(),
            "window_end": (issue + pd.Timedelta(days=horizon)).date().isoformat(),
            "threshold_mgL": threshold, "horizon_days": horizon,
            "label": label, "event_date": event_date,
            "reason": reason, "qc_suspect_issue": bool(suspect.loc[issue]),
        }
        for col in PHYSICAL_FEATURES:
            row[col] = float(d.at[issue, col]) if np.isfinite(d.at[issue, col]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def fit_logistic_ridge(train: pd.DataFrame, issue: pd.Series,
                       features: tuple[str, ...], alpha: float = 10.0):
    """Fixed-alpha L2 logistic regression with unpenalized intercept.

    Solver is Newton/IRLS with deterministic convergence, no hyperparameter
    search on future observations; returns one bounded event probability.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = train[list(features)].to_numpy(float)
    y = train["label"].to_numpy(float)
    q = issue[list(features)].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("Train and issue features must be finite")
    if not (0 < y.sum() < len(y)):
        raise ValueError("Both positive and negative classes required")
    mu, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (x - mu) / std
    q = (q - mu) / std
    z = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag([0.0] + [alpha] * len(features))
    beta = np.zeros(z.shape[1])
    beta[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) /
                     np.clip(1 - y.mean(), 1e-6, 1 - 1e-6))
    for _ in range(80):
        p = _sigmoid(z @ beta)
        var = np.maximum(p * (1 - p), 1e-6)
        gradient = z.T @ (y - p) - penalty @ beta
        hessian = z.T @ (z * var[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta += np.clip(step, -2.0, 2.0)
        if np.max(np.abs(step)) < 1e-7:
            break
    value = float(_sigmoid(beta[0] + q @ beta[1:]))
    return float(np.clip(value, 1e-6, 1 - 1e-6))


def backtest(daily: pd.DataFrame, thresholds=(4, 2), horizons=HORIZONS,
             months=None, normal_days=2, min_coverage=0.95,
             min_train=40, min_events=3, alpha=10.0):
    """Return risk forecasts, documented exclusions and label windows."""
    if not {"node", "date", "hours_lt4", "hours_lt2"}.issubset(daily):
        raise ValueError("Daily targets must contain node/date/hours_lt4/hours_lt2")
    if min_train < 2 or min_events < 1:
        raise ValueError("min_train must be >=2 and min_events >=1")
    allowed = None if months is None else set(str(m) for m in months)
    preds, exclusions, windows = [], [], []
    for node in sorted(daily["node"].unique()):
        for threshold in thresholds:
            for h in horizons:
                w = build_windows(
                    daily, node, threshold, h,
                    normal_days=normal_days, min_coverage=min_coverage)
                if w.empty:
                    continue
                windows.append(w[list(WINDOW_COLUMNS)])
                w["window_end"] = pd.to_datetime(w["window_end"])
                w["issue_timestamp"] = pd.to_datetime(w["issue_date"])
                # Incomplete target windows stay visible in window/exclusion
                # outputs but are never used as labels or training samples.
                tested = w if allowed is None else w.loc[
                    w["issue_date"].str[:7].isin(allowed)]
                for _, row in tested.iterrows():
                    issue_date = row["issue_timestamp"]
                    reason0 = (row["reason"] if row["reason"] != "evaluable"
                               else None)
                    history = w.loc[
                        w["label"].notna()
                        & w["window_end"].lt(issue_date)]
                    for model, features in MODEL_FEATURES.items():
                        reason = reason0
                        train = history
                        if reason is None and any(not np.isfinite(row[f])
                                                  for f in features):
                            reason = "missing_issue_features"
                        if features:
                            train = history.loc[
                                history[list(features)].notna().all(axis=1)]
                        count = len(train)
                        positives = int(train["label"].sum()) if count else 0
                        negatives = count - positives
                        if reason is None and count < min_train:
                            reason = "insufficient_training"
                        if (reason is None and features
                                and (positives < min_events
                                     or negatives < min_events)):
                            reason = "insufficient_event_classes"
                        exclusions.append({
                            "node": node, "issue_date": row["issue_date"],
                            "threshold_mgL": threshold, "horizon_days": h,
                            "model": model, "reason": reason or "evaluable",
                            "n_train": count, "n_train_events": positives,
                        })
                        if reason is not None:
                            continue
                        if not features:
                            probability = (positives + 1) / (count + 2)
                        else:
                            probability = fit_logistic_ridge(
                                train, row, features, alpha=alpha)
                        actual = int(row["label"])
                        preds.append({
                            "node": node, "issue_date": row["issue_date"],
                            "window_end": row["window_end"].date().isoformat(),
                            "event_date": row["event_date"],
                            "threshold_mgL": threshold, "horizon_days": h,
                            "model": model, "n_train": count,
                            "n_train_events": positives,
                            "observed_onset": actual,
                            "predicted_probability": float(probability),
                            "brier_error": (float(probability) - actual) ** 2,
                            "qc_suspect_issue": row["qc_suspect_issue"],
                        })
    return (
        pd.DataFrame(preds, columns=PRED_COLUMNS),
        pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS),
        pd.concat(windows, ignore_index=True)[list(WINDOW_COLUMNS)]
        if windows else pd.DataFrame(columns=WINDOW_COLUMNS),
    )


def average_precision(y_true, scores):
    """Non-interpolated precision-recall AP, correctly groups tied scores."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return np.nan
    order = np.argsort(-p, kind="stable")
    ranked, probabilities = y[order], p[order]
    boundaries = np.r_[np.flatnonzero(np.diff(probabilities)), len(y) - 1]
    n_seen = boundaries + 1
    tp = np.cumsum(ranked)[boundaries]
    recall = tp / y.sum()
    precision = tp / n_seen
    return float(np.diff(np.r_[0.0, recall]) @ precision)


def summarize(predictions: pd.DataFrame, by_station=False) -> pd.DataFrame:
    """Observed-event prevalence, Brier score and PR-average precision.

    No positives in an evaluation group => AP undefined (NaN), not zero.
    Report both positive and negative counts to avoid false confidence.
    """
    if predictions.empty:
        return pd.DataFrame()
    group = (["node"] if by_station else []) + [
        "threshold_mgL", "horizon_days", "model"]
    rows = []
    for key, g in predictions.groupby(group, sort=True):
        values = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        y = g["observed_onset"].to_numpy(dtype=int)
        p = g["predicted_probability"].to_numpy(dtype=float)
        values.update({
            "n_cases": len(g),
            "n_stations": g["node"].nunique(),
            "n_station_month_blocks": g.assign(
                month=g["issue_date"].str[:7]
            ).groupby(["node", "month"]).ngroups,
            "n_onset_events": int(y.sum()),
            "n_no_onset": int(len(y) - y.sum()),
            "onset_prevalence": float(y.mean()),
            "mean_predicted_risk": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "average_precision": average_precision(y, p),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def paired_risk_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare each learned model with a historical-rate baseline on SAME origins."""
    if predictions.empty:
        return pd.DataFrame()
    keys = ["node", "issue_date", "threshold_mgL", "horizon_days"]
    ref = predictions.loc[
        predictions["model"].eq("history_rate"),
        keys + ["predicted_probability", "observed_onset"],
    ].rename(columns={
        "predicted_probability": "rate_probability",
        "observed_onset": "rate_label",
    })
    joined = predictions.merge(ref, on=keys, how="inner", validate="many_to_one")
    if (joined["observed_onset"] != joined["rate_label"]).any():
        raise ValueError("Risk labels differ across models for matched issue")
    joined["historical_rate_brier"] = (
        joined["rate_probability"] - joined["observed_onset"]) ** 2
    rows = []
    for key, g in joined.groupby(
            ["threshold_mgL", "horizon_days", "model"]):
        rate = float(g["historical_rate_brier"].mean())
        model = float(g["brier_error"].mean())
        rows.append({
            "threshold_mgL": key[0], "horizon_days": key[1],
            "model": key[2], "n_paired": len(g),
            "n_onset_events": int(g["observed_onset"].sum()),
            "model_brier": model, "paired_history_rate_brier": rate,
            "brier_skill_vs_rate": (
                1 - model / rate if rate > 0 else np.nan),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Coverage-gated chronological hypoxia onset-risk benchmarks")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("trustworthy_outputs/onset"))
    parser.add_argument("--months", default="all",
                        help="Calendar issue months: all or 2025-09,2026-09")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--normal-days", type=int, default=2)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()
    d = pd.read_csv(args.daily)
    months = None if args.months.lower() == "all" else [
        s.strip() for s in args.months.split(",") if s.strip()]
    forecasts, exclusions, windows = backtest(
        d, months=months, normal_days=args.normal_days,
        min_coverage=args.min_coverage, min_train=args.min_train,
        min_events=args.min_events, alpha=args.alpha)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.out_dir / "onset_predictions.csv", index=False)
    windows.to_csv(args.out_dir / "onset_windows.csv", index=False)
    exclusions.groupby(
        ["node", "threshold_mgL", "horizon_days", "model", "reason"]
    ).size().rename("n_issues").reset_index().to_csv(
        args.out_dir / "onset_exclusions.csv", index=False)
    summarize(forecasts).to_csv(
        args.out_dir / "onset_summary_pooled.csv", index=False)
    summarize(forecasts, by_station=True).to_csv(
        args.out_dir / "onset_summary_station.csv", index=False)
    paired_risk_scores(forecasts).to_csv(
        args.out_dir / "onset_paired_brier.csv", index=False)
    print(f"Onset risk forecasts: {len(forecasts)} from "
          f"{forecasts.node.nunique()} stations; "
          f"{int(forecasts.observed_onset.sum())} positive forecast labels "
          f"(overlapping horizon windows may count events repeatedly)")
    print(f"Saved outputs to {args.out_dir}")
    if forecasts.empty:
        print("No eligible forecasts. Inspect onset_exclusions.csv. "
              "Sparse onset is a legitimate scientific result.")


if __name__ == "__main__":
    main()
