"""Time-valid, station-scoped DAILY environmental forcing for onset forecasts.

Input CSV is user-produced, not auto-fetched or retrospectively interpolated.
Every supplied measurement carries its observation end, availability time,
source ID, and local calendar day. A field can only enter issue-day features
once BOTH observation_end_utc and available_at_utc are <= end-of-issue-day.

Required columns:
  node,date,source_id,
  rain_mm,rain_observed_end_utc,rain_available_at_utc,rain_coverage,
  wind_mps,wind_observed_end_utc,wind_available_at_utc,wind_coverage,
  water_level_m,water_level_observed_end_utc,
  water_level_available_at_utc,water_level_coverage

Node must match an observed station exactly. "regional" is reserved for
regional rain/wind (not water levels); the analyst must explicitly elect to
broadcast those meteorological records. Any absent measurement is left NaN.
No backfill, climatology, spatial averaging, or future interpolation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/New_York"
FIELDS = ("rain_mm", "wind_mps", "water_level_m")
EXPECTED = ("node", "date", "source_id") + tuple(
    part for field in FIELDS
    for part in (field, f"{field}_observed_end_utc",
                 f"{field}_available_at_utc", f"{field}_coverage")
)
LIMITS = {
    "rain_mm": (0.0, 1000.0),
    "wind_mps": (0.0, 90.0),
    "water_level_m": (-10.0, 20.0),
}
MIN_COVERAGE = 0.90


def read_forcing(path: Path) -> pd.DataFrame:
    """Fail closed on malformed provenance; nonprovided fields remain null."""
    if path.suffix.lower() != ".csv":
        raise ValueError("Forcing input must be a real CSV, not a .url shortcut")
    d = pd.read_csv(path, dtype={"node": str, "source_id": str})
    absent = set(EXPECTED) - set(d)
    if absent:
        raise ValueError(f"Forcing CSV missing: {sorted(absent)}")
    if d.empty:
        raise ValueError("Forcing CSV empty")
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d",
                                errors="raise")
    if d["date"].isna().any() or d[["node", "source_id"]].isna().any().any():
        raise ValueError("Forcing date, node or source is missing")
    if d.duplicated(["node", "date"]).any():
        raise ValueError("Multiple sources for same node/day; consolidate explicitly")
    for field in FIELDS:
        d[field] = pd.to_numeric(d[field], errors="coerce")
        coverage = f"{field}_coverage"
        d[coverage] = pd.to_numeric(d[coverage], errors="coerce")
        for suffix in ("observed_end_utc", "available_at_utc"):
            column = f"{field}_{suffix}"
            d[column] = pd.to_datetime(d[column], utc=True, errors="coerce")
        valid = d[field].notna()
        if valid.any():
            lo, hi = LIMITS[field]
            if not np.isfinite(d.loc[valid, field]).all() or not d.loc[
                valid, field].between(lo, hi).all():
                raise ValueError(f"{field} outside plausible unit bounds")
            if (d.loc[valid, coverage].isna().any()
                    or not d.loc[valid, coverage].between(0, 1).all()):
                raise ValueError(f"Missing/invalid coverage for {field}")
            start = d.loc[valid, "date"].dt.tz_localize(
                LOCAL_TZ).dt.tz_convert("UTC")
            end = (d.loc[valid, "date"] + pd.DateOffset(days=1)
                   ).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")
            observed = d.loc[valid, f"{field}_observed_end_utc"]
            available = d.loc[valid, f"{field}_available_at_utc"]
            # Require end of daily aggregate, not a morning snapshot passed
            # off as complete day. This also handles DST lengths correctly.
            if (observed.isna().any() or available.isna().any()
                    or (observed < start).any()
                    or (observed > end).any()
                    or (available < observed).any()):
                raise ValueError(f"Invalid timestamps for {field}")
        # Absent fields cannot masquerade as having a usable observation.
        missing = ~valid
        if (d.loc[missing, coverage].notna().any()
                or d.loc[missing, f"{field}_observed_end_utc"].notna().any()
                or d.loc[missing, f"{field}_available_at_utc"].notna().any()):
            raise ValueError(f"{field}: missing value has nonmissing provenance")
    return d


def materialize(daily: pd.DataFrame, forcing: pd.DataFrame,
                min_coverage: float = MIN_COVERAGE,
                allow_regional_weather: bool = False) -> pd.DataFrame:
    """Return issue-date feature rows without future/unknown meteorology.

    For regional weather, caller explicitly opts in; never broadcast regional
    water level. Node-specific sources override opted-in regional weather.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")
    if not {"node", "date"}.issubset(daily):
        raise ValueError("Daily data must contain node and local date")
    d = daily[["node", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    if d.duplicated(["node", "date"]).any():
        raise ValueError("Duplicate station daily dates")
    if forcing.duplicated(["node", "date"]).any():
        raise ValueError("Duplicate forcing station/day")
    f = forcing.copy()
    direct = d.merge(f, on=["node", "date"], how="left",
                     validate="one_to_one")
    if allow_regional_weather:
        region = f.loc[f["node"].eq("regional")].drop(columns=["node"])
        regional = d[["date"]].merge(
            region, on="date", how="left", validate="many_to_one")
    else:
        regional = None
    result = d.copy()
    issue_end = (d["date"] + pd.DateOffset(days=1)).dt.tz_localize(
        LOCAL_TZ).dt.tz_convert("UTC")
    for field in FIELDS:
        good = (
            direct[field].notna()
            & direct[f"{field}_coverage"].ge(min_coverage)
            & direct[f"{field}_available_at_utc"].le(issue_end)
            & direct[f"{field}_observed_end_utc"].le(issue_end)
        )
        vals = direct[field].where(good)
        if regional is not None and field in ("rain_mm", "wind_mps"):
            local_missing = direct[field].isna()
            # Regional weather applies only if station-specific data is
            # ABSENT, never if it exists but arrives too late or is bad.
            regional_ok = (
                regional[field].notna()
                & regional[f"{field}_coverage"].ge(min_coverage)
                & regional[f"{field}_available_at_utc"].le(issue_end)
                & regional[f"{field}_observed_end_utc"].le(issue_end)
            )
            vals = vals.where(~(local_missing & regional_ok),
                              regional[field].where(local_missing & regional_ok))
        result[field] = vals.to_numpy(dtype=float)
    result["forcing_complete"] = result[list(FIELDS)].notna().all(axis=1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit time-valid forcing observations at each issue day")
    parser.add_argument("--daily", type=Path,
                        default=Path("trustworthy_outputs/daily_targets.csv"))
    parser.add_argument("--forcing", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(
        "trustworthy_outputs/onset/time_valid_forcing.csv"))
    parser.add_argument("--allow-regional-weather", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = parser.parse_args()
    daily = pd.read_csv(args.daily)
    forcing = read_forcing(args.forcing)
    out = materialize(daily, forcing, args.min_coverage,
                      args.allow_regional_weather)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Time-valid complete rain/wind/water-level station-days: "
          f"{out.forcing_complete.sum()}/{len(out)}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
