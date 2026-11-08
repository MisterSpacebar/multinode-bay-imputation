"""Observed-only daily hypoxia episodes and optional strict five-minute detail.

Daily episodes are consecutive *eligible local days* with at least one observed
sample below a threshold. They are distinct from continuous five-minute events:
multiple subdaily excursions may occur inside one hypoxic calendar day.

Unknown days, missing dates and record boundaries censor episodes; none imply
recovery. This module does not fit models, bridge missing data or use GNN values
as observed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.quality import read_station
from trustworthy.targets import detect_hypoxia_events

EPISODE_COLUMNS = [
    "episode_id", "node", "threshold_mgL", "start_date_local",
    "last_hypoxic_date_local", "recovery_date_local", "calendar_duration_days",
    "hours_below_threshold", "eligible_coverage_min", "censored_left",
    "censored_right", "observed_recovery", "left_boundary", "right_boundary",
    "identifiable_onset", "inter_onset_days", "nonhypoxic_days_since_prior_run",
    "season", "qc_suspect", "n_qc_suspect_days", "provenance",
]
SUBDAILY_COLUMNS = [
    "episode_id", "node", "threshold_mgL", "start_utc", "end_utc",
    "onset_local", "observed_duration_minutes", "hours_below_threshold",
    "n_observed_samples", "min_do_mgL", "mean_do_mgL",
    "do_deficit_mgL_hours", "censored_left", "censored_right",
    "qc_suspect", "provenance",
]
SUMMARY_COLUMNS = [
    "node", "threshold_mgL", "n_observed_runs", "n_identifiable_onsets",
    "n_left_censored", "n_recovered_onsets", "n_right_censored_onsets",
    "n_qc_suspect_runs", "median_onset_run_days", "max_onset_run_days",
    "median_onset_exposure_hours",
]


def _bool(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string").str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    # Reindexed calendar gaps have NA flags; treat as unknown rather than False.
    original = values.notna()
    if result.loc[original].isna().any():
        raise ValueError(f"{name} has invalid boolean flags")
    return result.fillna(False).astype(bool)


def _season(date: pd.Timestamp) -> str:
    month = date.month
    return ("DJF" if month in (12, 1, 2) else
            "MAM" if month in (3, 4, 5) else
            "JJA" if month in (6, 7, 8) else "SON")


def _runs(mask: np.ndarray):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return
    for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        yield int(part[0]), int(part[-1]) + 1


def daily_episode_table(regimes: pd.DataFrame, thresholds=(2, 4),
                        min_coverage: float = 0.95) -> pd.DataFrame:
    """Extract one row per uninterrupted eligible daily hypoxic run.

    A new onset is identifiable only when the immediately preceding calendar
    day is eligible and non-hypoxic. An observed recovery requires the next
    calendar day to be eligible and non-hypoxic. The reported exposure is a
    lower bound on hours below threshold whenever daily coverage is <100%.
    """
    needed = {"node", "date", "threshold_mgL", "eligible", "hours_below", "coverage"}
    missing = needed - set(regimes.columns)
    if missing:
        raise ValueError(f"regime_days missing columns: {sorted(missing)}")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0,1]")
    if any(float(t) not in (2.0, 4.0) for t in thresholds):
        raise ValueError("Only observed 2 and 4 mg/L thresholds are supported")
    d = regimes.copy()
    d["node"] = d["node"].astype("string")
    if d["node"].isna().any():
        raise ValueError("Missing station identity")
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    if d["date"].isna().any() or d["date"].dt.tz is not None:
        raise ValueError("Daily dates must be local, timezone-naive dates")
    if (d["date"].dt.normalize() != d["date"]).any():
        raise ValueError("Daily dates must be midnight local calendar dates")
    for col in ("threshold_mgL", "hours_below", "coverage"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d[["threshold_mgL"]].isna().any().any():
        raise ValueError("Missing threshold")
    if d.duplicated(["node", "date", "threshold_mgL"]).any():
        raise ValueError("Duplicate station/date/threshold in regime file")

    rows = []
    for (node, threshold), group in d.groupby(["node", "threshold_mgL"],
                                               sort=True):
        if threshold not in thresholds:
            continue
        group = group.set_index("date").sort_index()
        group = group.reindex(pd.date_range(group.index.min(),
                                             group.index.max(), freq="D"))
        eligible = _bool(group["eligible"], "eligible")
        coverage = group["coverage"]
        hours = group["hours_below"]
        valid = (eligible & coverage.ge(min_coverage) & hours.notna()
                 & np.isfinite(hours) & hours.between(0, 24))
        active = (valid & hours.gt(0)).to_numpy(dtype=bool)
        valid_array = valid.to_numpy(dtype=bool)
        suspect = (_bool(group["qc_suspect"], "qc_suspect")
                   if "qc_suspect" in group else
                   pd.Series(False, index=group.index))
        previous = None
        for first, end in _runs(active):
            dates = group.index
            left_ok = first > 0 and valid_array[first - 1]
            right_ok = end < len(group) and valid_array[end]
            # Runs are maximal; an eligible adjacent day is non-hypoxic.
            left_censored = not left_ok
            right_censored = not right_ok
            onset = dates[first]
            final = dates[end - 1]
            n_days = end - first
            assert n_days >= 1
            row = {
                "episode_id": f"{node}::{int(threshold)}::{onset.date().isoformat()}",
                "node": str(node),
                "threshold_mgL": int(threshold),
                "start_date_local": onset.date().isoformat(),
                "last_hypoxic_date_local": final.date().isoformat(),
                "recovery_date_local": (dates[end].date().isoformat()
                                        if right_ok else None),
                "calendar_duration_days": n_days,
                "hours_below_threshold": float(hours.iloc[first:end].sum()),
                "eligible_coverage_min": float(coverage.iloc[first:end].min()),
                "censored_left": left_censored,
                "censored_right": right_censored,
                "observed_recovery": right_ok,
                "left_boundary": ("record_start" if first == 0 else
                                  "unknown_day" if left_censored else "normal_day"),
                "right_boundary": ("record_end" if end == len(group) else
                                   "unknown_day" if right_censored else "normal_day"),
                "identifiable_onset": not left_censored,
                "inter_onset_days": np.nan,
                "nonhypoxic_days_since_prior_run": np.nan,
                "season": _season(onset),
                "qc_suspect": bool(suspect.iloc[first:end].any()),
                "n_qc_suspect_days": int(suspect.iloc[first:end].sum()),
                "provenance": "observed_only",
            }
            # Recurrence stays within station and threshold. Do not use a
            # left-censored start as if its actual onset date were known.
            if previous is not None and not left_censored:
                if previous["identifiable_onset"]:
                    prev_start = pd.Timestamp(previous["start_date_local"])
                    row["inter_onset_days"] = int((onset - prev_start).days)
                if previous["observed_recovery"]:
                    prev_end = pd.Timestamp(previous["last_hypoxic_date_local"])
                    row["nonhypoxic_days_since_prior_run"] = (
                        int((onset - prev_end).days) - 1)
            rows.append(row)
            previous = row
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return station-threshold descriptors, without making KM claims."""
    rows = []
    if episodes.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for (node, threshold), g in episodes.groupby(
            ["node", "threshold_mgL"], sort=True):
        identified = g.loc[g["identifiable_onset"].astype(bool)]
        rows.append({
            "node": node, "threshold_mgL": threshold,
            "n_observed_runs": len(g),
            "n_identifiable_onsets": len(identified),
            "n_left_censored": int(g["censored_left"].sum()),
            "n_recovered_onsets": int(identified["observed_recovery"].sum()),
            "n_right_censored_onsets": int(identified["censored_right"].sum()),
            "n_qc_suspect_runs": int(g["qc_suspect"].sum()),
            "median_onset_run_days": (
                float(identified["calendar_duration_days"].median())
                if not identified.empty else np.nan),
            "max_onset_run_days": (
                int(identified["calendar_duration_days"].max())
                if not identified.empty else np.nan),
            "median_onset_exposure_hours": (
                float(identified["hours_below_threshold"].median())
                if not identified.empty else np.nan),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def subdaily_episode_table(input_dir: Path, interval_minutes: int = 5) -> pd.DataFrame:
    """Add exact metrics from source-flagged five-minute exports, if available.

    Uses trustworthy.read_station (observed flags required) and the existing
    strict five-minute event detector. Never reads reconstructed values as DO.
    This table is intentionally separate from daily episodes.
    """
    manifest = input_dir / "node_provenance.csv"
    files = sorted(input_dir.glob("*_imputed.csv"))
    if manifest.exists():
        names = pd.read_csv(manifest)["node"].astype(str).tolist()
        if not names or len(names) != len(set(names)):
            raise ValueError("Invalid node_provenance.csv")
        files = [input_dir / f"{name}_imputed.csv" for name in names]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Manifest exports missing: {missing[:5]}")
    else:
        print("WARNING: no node_provenance.csv; scanning all *_imputed.csv files")
    if not files:
        raise ValueError(f"No source-flagged exports in {input_dir}")
    rows = []
    for file in files:
        node = file.name.removesuffix("_imputed.csv")
        qc = read_station(file)
        for event in detect_hypoxia_events(qc, node,
                                           interval_minutes=interval_minutes):
            first, end = pd.Timestamp(event["start_utc"]), pd.Timestamp(event["end_utc"])
            block = qc.loc[(qc.index >= first) & (qc.index < end)]
            observed = block["do_valid"].dropna().astype(float)
            expected = event["observed_duration_minutes"] // interval_minutes
            if len(observed) != expected:
                raise ValueError("Detected event contains a missing/invalid sample")
            threshold = float(event["threshold_mgL"])
            suspect = any(bool(block[c].any()) for c in (
                "do_flatline", "do_spike_suspect", "do_zero_review") if c in block)
            rows.append({
                "episode_id": f"{node}::{int(threshold)}::{first.isoformat()}",
                "node": node, "threshold_mgL": int(threshold),
                "start_utc": first.isoformat(), "end_utc": end.isoformat(),
                "onset_local": first.tz_convert("America/New_York").isoformat(),
                "observed_duration_minutes": event["observed_duration_minutes"],
                "hours_below_threshold": event["observed_duration_minutes"] / 60.0,
                "n_observed_samples": len(observed),
                "min_do_mgL": float(observed.min()),
                "mean_do_mgL": float(observed.mean()),
                "do_deficit_mgL_hours": float(
                    (threshold - observed).clip(lower=0).sum()
                    * interval_minutes / 60),
                "censored_left": bool(event["censored_left"]),
                "censored_right": bool(event["censored_right"]),
                "qc_suspect": suspect, "provenance": "observed_only",
            })
    return pd.DataFrame(rows, columns=SUBDAILY_COLUMNS)


def make_figures(episodes: pd.DataFrame, destination: Path) -> None:
    """Optional exploratory figures (matplotlib is not required by CI)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib not installed; figure generation skipped")
        return
    destination.mkdir(parents=True, exist_ok=True)
    onsets = episodes.loc[episodes["identifiable_onset"]]
    if onsets.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for threshold, subset in onsets.groupby("threshold_mgL"):
        ax.hist(subset["calendar_duration_days"], alpha=0.45, bins=20,
                label=f"DO < {threshold:g} mg/L")
    ax.set(xlabel="Observed consecutive eligible hypoxic days",
           ylabel="Onset-defined episodes",
           title="Observed episode lengths (right-censored runs included)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "episode_duration_histogram.png", dpi=160)
    plt.close(fig)

    station = onsets.groupby(["node", "threshold_mgL"]).size().unstack(fill_value=0)
    ax = station.plot(kind="bar", figsize=(10, 5))
    ax.set(ylabel="Identifiable observed onsets",
           title="Hypoxia onsets by station (varying observation windows)")
    ax.figure.tight_layout()
    ax.figure.savefig(destination / "episode_station_summary.png", dpi=160)
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed-only hypoxia episodes")
    parser.add_argument("--regimes", type=Path,
                        default=Path("trustworthy_outputs/onset/regime_days.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("trustworthy_outputs/episode_dynamics"))
    parser.add_argument("--thresholds", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--exports-dir", type=Path, default=None,
                        help="Optional source-flagged 5-min exports; no GNN substitution")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if not args.regimes.exists():
        parser.error(f"Missing {args.regimes}; run python -m trustworthy.regimes first")
    episodes = daily_episode_table(pd.read_csv(args.regimes),
                                   thresholds=args.thresholds,
                                   min_coverage=args.min_coverage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out_dir / "episode_table.csv", index=False)
    summary = summarize_episodes(episodes)
    summary.to_csv(args.out_dir / "station_episode_summary.csv", index=False)
    totals = (summary.groupby("threshold_mgL", as_index=False)[[
        "n_observed_runs", "n_identifiable_onsets", "n_left_censored",
        "n_recovered_onsets", "n_right_censored_onsets", "n_qc_suspect_runs",
    ]].sum() if not summary.empty else pd.DataFrame())
    totals.to_csv(args.out_dir / "episode_summary.csv", index=False)
    episodes.groupby(["threshold_mgL", "season"], dropna=False).agg(
        n_observed_runs=("episode_id", "size"),
        n_identifiable_onsets=("identifiable_onset", "sum"),
    ).reset_index().to_csv(args.out_dir / "episode_season_summary.csv", index=False)
    if args.exports_dir is not None:
        detail = subdaily_episode_table(args.exports_dir)
        detail.to_csv(args.out_dir / "subdaily_episodes.csv", index=False)
    if not args.no_figures:
        make_figures(episodes, args.out_dir / "figures")
    metadata = {
        "regimes_path": str(args.regimes),
        "regimes_sha256": hashlib.sha256(args.regimes.read_bytes()).hexdigest(),
        "thresholds_mgL": args.thresholds,
        "min_coverage": args.min_coverage,
        "n_daily_runs": len(episodes),
        "daily_definition": "consecutive eligible local days with any observed below-threshold sample",
        "subdaily_definition": "continuous observed 5-minute below-threshold samples",
        "provenance": "observed_only",
        "censoring": "unknown/gap/record boundary is never observed recovery",
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(episodes)} daily observed runs to {args.out_dir}")


if __name__ == "__main__":
    main()
