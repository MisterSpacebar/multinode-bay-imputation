"""Observed-only station QC and daily hypoxia targets.

Reads the existing *_imputed.csv exports but NEVER uses their imputed values.
Run from the repository root: python -m trustworthy.quality --help

This is a retrospective audit, not a real-time alerting system.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    interval_minutes: int = 5
    min_coverage: float = 0.95
    flatline_samples: int = 12
    local_tz: str = "America/New_York"

    def __post_init__(self):
        if self.interval_minutes <= 0 or 1440 % self.interval_minutes:
            raise ValueError("interval_minutes must divide 1440")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        if self.flatline_samples < 2:
            raise ValueError("flatline_samples must be at least 2")


BOUNDS = {
    "odo_mgL": (0.0, 20.0),
    "temp_c": (-5.0, 45.0),
    "sal_ppt": (0.0, 45.0),
}


def observed_flag(values: pd.Series) -> pd.Series:
    """Reject missing/ambiguous provenance instead of interpreting strings as bool."""
    mapped = values.astype("string").str.strip().str.lower().map(
        {"1": True, "0": False, "true": True, "false": False}
    )
    if mapped.isna().any():
        raise ValueError("Observation flag contains missing or non-boolean values")
    return mapped.astype(bool)


def read_station(path: Path, config: Config = Config()) -> pd.DataFrame:
    """Return a UTC-indexed QC frame. Naive timestamps are assumed UTC.

    *_observed flags are mandatory for DO, temperature and salinity; optional
    physical features without their own flags are not used as observed inputs.
    """
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path}: empty file")
    time_col = "datetime" if "datetime" in frame else frame.columns[0]
    time = pd.to_datetime(frame[time_col], utc=True, errors="raise")
    if time.isna().any():
        raise ValueError(f"{path}: missing timestamps")
    frame.index = pd.DatetimeIndex(time)
    if frame.index.has_duplicates:
        raise ValueError(f"{path}: duplicate timestamps; resolve upstream")
    frame = frame.sort_index()

    if "odo_mgL" not in frame or "odo_mgL_observed" not in frame:
        raise ValueError(f"{path}: DO and its observation flag are required")

    out = pd.DataFrame(index=frame.index)
    for feature, (lower, upper) in BOUNDS.items():
        flag_col = f"{feature}_observed"
        if feature not in frame or flag_col not in frame:
            if feature == "odo_mgL":
                raise ValueError(f"{path}: missing observed DO")
            out[feature] = np.nan
            continue
        is_observed = observed_flag(frame[flag_col])
        raw = pd.to_numeric(frame[feature], errors="coerce")
        valid_range = raw.notna() & np.isfinite(raw) & raw.between(lower, upper)
        out[feature] = raw.where(is_observed & valid_range)
        if feature == "odo_mgL":
            out["do_flag_observed"] = is_observed
            out["do_invalid"] = is_observed & ~valid_range

    # Retrospective QC: flag the ENTIRE run if 12 consecutive grid-aligned
    # observed readings are exactly identical. No interpolation across gaps.
    do = out["odo_mgL"]
    step = pd.Timedelta(minutes=config.interval_minutes)
    contiguous = out.index.to_series().diff().eq(step)
    new_run = do.ne(do.shift()) | ~contiguous | do.isna()
    run_id = new_run.cumsum()
    run_size = do.groupby(run_id).transform("size")
    flatline = do.notna() & run_size.ge(config.flatline_samples)
    out["do_flatline"] = flatline
    out["do_zero_review"] = do.eq(0)  # zero can be genuine; review, do not auto-delete
    out["do_valid"] = do.where(~flatline)
    return out


def read_deployments(path: Path | None) -> pd.DataFrame | None:
    """Optional pseudonymous instrument epochs: node,start_utc,end_utc,deployment_id.

    end_utc is exclusive. Actual data-source names are not required.
    """
    if path is None:
        return None
    df = pd.read_csv(path, dtype={"node": str, "deployment_id": str})
    required = {"node", "start_utc", "end_utc", "deployment_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Deployment map needs columns: {sorted(required)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Deployment map has null fields")
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True)
    if (df["end_utc"] <= df["start_utc"]).any():
        raise ValueError("Deployment end must be after start")
    for node, group in df.groupby("node"):
        ordered = group.sort_values("start_utc")
        if (ordered["start_utc"].iloc[1:].array < ordered["end_utc"].iloc[:-1].array).any():
            raise ValueError(f"Overlapping deployments for {node}")
    return df


def deployment_labels(index: pd.DatetimeIndex, node: str,
                      deployments: pd.DataFrame | None) -> pd.Series:
    labels = pd.Series("unverified", index=index, dtype="string")
    if deployments is None:
        return labels
    for _, item in deployments.loc[deployments["node"] == node].iterrows():
        selection = (index >= item["start_utc"]) & (index < item["end_utc"])
        labels.loc[selection] = item["deployment_id"]
    return labels


def expected_slots(day: pd.Timestamp, config: Config) -> int:
    """Local-day duration is 23/24/25 h across daylight-saving transitions."""
    midnight = pd.Timestamp(day.date()).tz_localize(config.local_tz)
    next_midnight = midnight + pd.DateOffset(days=1)
    return int((next_midnight - midnight) / pd.Timedelta(minutes=config.interval_minutes))


def daily_targets(qc: pd.DataFrame, node: str, config: Config = Config(),
                  deployments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Strict daily labels: NaN when valid DO coverage is below the threshold.

    hours_lt4 and hours_lt2 are OBSERVED hours (lower bounds if coverage < 100%),
    not extrapolations to the missing portions of a day.
    """
    local = qc.tz_convert(config.local_tz).copy()
    local["deployment_id"] = deployment_labels(qc.index, node, deployments).to_numpy()
    local["local_day"] = local.index.date
    rows = []
    for day, group in local.groupby("local_day", sort=True):
        date = pd.Timestamp(day)
        expected = expected_slots(date, config)
        if len(group) > expected:
            raise ValueError(f"{node} {date.date()}: more readings than expected grid slots")
        good = group["do_valid"].dropna()
        coverage = len(good) / expected
        eligible = coverage >= config.min_coverage
        deployments_today = group.loc[group["do_valid"].notna(), "deployment_id"].unique()
        epoch = deployments_today[0] if len(deployments_today) == 1 else "mixed"
        if epoch == "mixed":
            eligible = False
        predawn = group[(group.index.hour >= 0) & (group.index.hour < 6)]["do_valid"]
        predawn_expected = int(6 * 60 / config.interval_minutes)
        predawn_min = (predawn.min() if predawn.notna().sum() >=
                       int(np.ceil(0.9 * predawn_expected)) else np.nan)
        rows.append({
            "node": node,
            "date": date.date().isoformat(),
            "deployment_id": str(epoch),
            "source_verified": epoch not in ("unverified", "mixed"),
            "eligible": bool(eligible),
            "expected_slots": expected,
            "observed_slots": int(group["do_flag_observed"].sum()),
            "valid_slots": int(len(good)),
            "coverage": coverage,
            "n_invalid": int(group["do_invalid"].sum()),
            "n_flatline": int(group["do_flatline"].sum()),
            "n_zero_review": int(group["do_zero_review"].sum()),
            "odo_mean": float(good.mean()) if eligible else np.nan,
            "odo_min": float(good.min()) if eligible else np.nan,
            "odo_range": float(good.max() - good.min()) if eligible else np.nan,
            "predawn_min": float(predawn_min) if eligible else np.nan,
            "hours_lt4": float((good < 4.0).sum() * config.interval_minutes / 60)
                         if eligible else np.nan,
            "hours_lt2": float((good < 2.0).sum() * config.interval_minutes / 60)
                         if eligible else np.nan,
            "hypoxia_detected": int((good < 4.0).any()) if eligible else np.nan,
            "temp_mean": float(group["temp_c"].mean()) if eligible else np.nan,
            "sal_mean": float(group["sal_ppt"].mean()) if eligible else np.nan,
        })
    return pd.DataFrame(rows)


def station_audit(qc: pd.DataFrame, node: str) -> dict:
    return {
        "node": node,
        "first_utc": qc.index.min().isoformat(),
        "last_utc": qc.index.max().isoformat(),
        "n_rows": len(qc),
        "n_do_observed_flag": int(qc["do_flag_observed"].sum()),
        "n_do_invalid": int(qc["do_invalid"].sum()),
        "n_do_flatline": int(qc["do_flatline"].sum()),
        "n_do_zero_review": int(qc["do_zero_review"].sum()),
        "n_do_valid": int(qc["do_valid"].notna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed-only DO QC and daily targets")
    parser.add_argument("--input-dir", type=Path, default=Path("imputed_output"))
    parser.add_argument("--out-dir", type=Path, default=Path("trustworthy_outputs"))
    parser.add_argument("--deployments", type=Path, default=None,
                        help="Optional pseudonymous station/deployment epoch CSV")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--flatline-samples", type=int, default=12)
    args = parser.parse_args()
    config = Config(args.interval_minutes, args.min_coverage, args.flatline_samples)
    deployments = read_deployments(args.deployments)
    paths = sorted(args.input_dir.glob("*_imputed.csv"))
    if not paths:
        parser.error(f"No *_imputed.csv files in {args.input_dir}")
    summaries, audits = [], []
    for path in paths:
        node = path.name.removesuffix("_imputed.csv")
        qc = read_station(path, config)
        summaries.append(daily_targets(qc, node, config, deployments))
        audits.append(station_audit(qc, node))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(summaries, ignore_index=True)
    daily.to_csv(args.out_dir / "daily_targets.csv", index=False)
    pd.DataFrame(audits).to_csv(args.out_dir / "station_audit.csv", index=False)
    print(f"Audited {len(paths)} stations; {int(daily['eligible'].sum())} eligible station-days")
    print(f"Saved {args.out_dir / 'daily_targets.csv'}")
    if deployments is None:
        print("WARNING: deployment epochs unverified; cross-instrument comparisons remain exploratory.")


if __name__ == "__main__":
    main()
