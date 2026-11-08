"""Build source-auditable daily forcing for retrospective hypoxia analysis.

Source location and interpretation:
- Wind: Virginia Key NDBC VAKF1, followed by Virginia Key CO-OPS 8723214
  wind observations. Assemble timestamp-level series BEFORE aggregating by
  America/New_York local day, preserving mixed-source handoff provenance.
- Water level: CO-OPS 8723214 coastal boundary, MSL datum, meters. Its daily
  mean is NOT a local canal/bay sensor or a tidal-range estimate.
- Rainfall: MIA airport daily totals (inches -> millimeters). The source CSV
  confirms a daily total was recorded, but does NOT verify subdaily coverage.

Values are broadcast to observed sensor nodes as explicit *regional proxies*;
17 node rows do NOT imply 17 independent meteorological instruments.
All source identities and coverage bases are retained as columns.

Publication times are NOT provided by the historical files. The 3-hour
weather/level and 24-hour rainfall lags are stated assumptions only, NOT
verified arrival times. Forecast ablations must retain this qualification.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/New_York"
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION = "8723214"
SLOTS_PER_HOUR = 10  # six-minute observations; 23/25-hour DST days handled


def fetch_coops(product: str, start: pd.Timestamp, end: pd.Timestamp,
                cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=30), end)
        path = cache / f"{product}_{cur:%Y%m%d}_{stop:%Y%m%d}.csv"
        if not path.exists():
            url = (f"{API}?product={product}&application=bay_research"
                   f"&begin_date={cur:%Y%m%d}&end_date={stop:%Y%m%d}"
                   f"&station={STATION}&time_zone=gmt&units=metric&format=csv")
            if product == "water_level":
                url += "&datum=MSL"
            with urllib.request.urlopen(url, timeout=120) as response:
                response_text = response.read().decode("utf-8")
            if response_text.lstrip().startswith("Error") or "Date Time" not in response_text:
                raise RuntimeError(
                    f"CO-OPS {product} {cur:%Y-%m-%d}: {response_text[:200]}")
            path.write_text(response_text, encoding="utf-8")
        frames.append(pd.read_csv(path, skipinitialspace=True))
        cur = stop + pd.Timedelta(days=1)
    d = pd.concat(frames, ignore_index=True)
    d.columns = [c.strip() for c in d.columns]
    d["time"] = pd.to_datetime(d["Date Time"], utc=True)
    return d.drop_duplicates("time").set_index("time").sort_index()


def read_ndbc(folder: Path) -> pd.Series:
    parts = []
    for p in sorted(folder.glob("vakf1h*.txt")):
        d = pd.read_csv(
            p, sep=r"\s+", comment="#", header=None, usecols=range(7),
            names=["yy", "mo", "dy", "hh", "mm", "wdir", "wspd"])
        t = pd.to_datetime(dict(
            year=d.yy, month=d.mo, day=d.dy, hour=d.hh, minute=d.mm),
            utc=True)
        s = pd.Series(d.wspd.to_numpy(float), index=t)
        parts.append(s.where(s.between(0, 90, inclusive="left")))
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


def splice_wind(ndbc: pd.Series, coops: pd.Series):
    """Join by UTC timestamp before grouping into local calendar days.

    Prefer NDBC up to the last *valid* NDBC observation, then CO-OPS.
    This fills the UTC/new-year versus Miami-local-day seam without
    double-counting overlapping source timestamps.
    """
    a = pd.to_numeric(ndbc, errors="coerce").dropna().sort_index()
    b = pd.to_numeric(coops, errors="coerce")
    b = b.where(b.between(0, 90, inclusive="left")).dropna().sort_index()
    if a.empty and b.empty:
        raise ValueError("No valid NDBC or CO-OPS wind samples")
    if not a.empty:
        b = b.loc[b.index > a.index.max()]
    s = pd.concat([a, b]).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    origins = pd.concat([
        pd.Series("ndbc_vakf1", index=a.index),
        pd.Series("coops_8723214_wind", index=b.index),
    ]).reindex(s.index)
    return s, origins


def daily_aggregate(series: pd.Series, lag_hours: float, source_id: str,
                    name: str, origins: pd.Series | None = None) -> pd.DataFrame:
    """Aggregate complete local days with observation-derived slot coverage."""
    if lag_hours < 0:
        raise ValueError("Assumed availability lag cannot be negative")
    s = series.dropna().sort_index()
    if s.empty:
        raise ValueError(f"No valid observations for {name}")
    local_date = s.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    g = s.groupby(local_date)
    out = pd.DataFrame({
        name: g.mean(), "n": g.size(),
        "last": g.apply(lambda x: x.index.max()),
    })
    day_start = out.index.tz_localize(LOCAL_TZ).tz_convert("UTC")
    day_end = (out.index + pd.DateOffset(days=1)).tz_localize(
        LOCAL_TZ).tz_convert("UTC")
    hours = (day_end - day_start).total_seconds() / 3600
    out[f"{name}_coverage"] = (out["n"] / (
        hours * SLOTS_PER_HOUR)).clip(upper=1.0)
    end = (out["last"] + pd.Timedelta(minutes=6)).clip(
        upper=pd.Series(day_end, index=out.index))
    out[f"{name}_observed_end_utc"] = end
    out[f"{name}_available_at_utc"] = end + pd.Timedelta(hours=lag_hours)
    if origins is None:
        out[f"{name}_source"] = source_id
    else:
        labels = origins.reindex(s.index)
        if labels.isna().any():
            raise ValueError(f"Missing observation source labels for {name}")
        # Preserve the order first encountered on this local day.
        out[f"{name}_source"] = labels.groupby(local_date).agg(
            lambda x: "+".join(pd.unique(x.astype(str))))
    out.index.name = "date"
    return out.drop(columns=["n", "last"])


def read_rain(folder: Path, lag_hours: float) -> pd.DataFrame:
    """MIA daily total: record presence is known, intra-day sampling is not."""
    if lag_hours < 0:
        raise ValueError("Assumed rainfall publication lag cannot be negative")
    files = [folder / "rainfall_2025_mar_oct_combined.csv",
             *sorted(folder.glob("miami_airport_daily_2026_*.csv"))]
    existing = [f for f in files if f.exists()]
    if not existing:
        raise FileNotFoundError(f"No MIA rainfall CSVs under {folder}")
    d = pd.concat([pd.read_csv(f) for f in existing], ignore_index=True)
    d["date"] = pd.to_datetime(dict(
        year=d.year, month=d.month, day=d.day))
    d = d.drop_duplicates("date").set_index("date").sort_index()
    out = pd.DataFrame({
        "rain_mm": pd.to_numeric(d["rain_in"], errors="coerce") * 25.4})
    out = out.dropna()
    end = (out.index + pd.DateOffset(days=1)).tz_localize(
        LOCAL_TZ).tz_convert("UTC")
    # Existing validator accepts a 0-1 completeness field. Here 1 means
    # "daily archive record present", NOT independently verified 24h coverage.
    # Keep that semantics explicit in the additional basis column.
    out["rain_mm_coverage"] = 1.0
    out["rain_mm_observed_end_utc"] = end
    out["rain_mm_available_at_utc"] = end + pd.Timedelta(hours=lag_hours)
    out["rain_mm_source"] = "mia_airport_daily"
    out.index.name = "date"
    return out


def build_wind(ndbc: pd.Series, first: pd.Timestamp,
               last: pd.Timestamp, cache: Path, lag_hours: float) -> pd.DataFrame:
    """Retrieve a UTC overlap around the switch, then splice before daily aggregation."""
    valid = ndbc.dropna()
    if valid.empty:
        # No NDBC files: use CO-OPS from the beginning with UTC padding.
        coops_start = first - pd.Timedelta(days=2)
    else:
        # Query the LAST NDBC UTC date, not the following day, because
        # the final UTC hours are still the preceding Miami calendar day.
        coops_start = valid.index.max().tz_convert(None).normalize()
    if coops_start <= last:
        coops = fetch_coops("wind", coops_start, last, cache)["Speed"].astype(float)
    else:
        coops = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    wind, origin = splice_wind(ndbc, coops)
    return daily_aggregate(
        wind, lag_hours, "mixed_ndbc_coops", "wind_mps", origins=origin)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rain-dir", type=Path, default=Path("water_data/rainfall"))
    p.add_argument("--rain-availability-lag-hours", type=float, default=24.0)
    p.add_argument("--daily", type=Path,
                   default=Path("trustworthy_outputs/daily_targets.csv"))
    p.add_argument("--wind-dir", type=Path, default=Path("water_data/wind_data"))
    p.add_argument("--cache", type=Path,
                   default=Path("water_data/coops_8723214"))
    p.add_argument("--out", type=Path, default=Path("water_data/forcing_daily.csv"))
    p.add_argument("--availability-lag-hours", type=float, default=3.0)
    a = p.parse_args()

    targets = pd.read_csv(a.daily, parse_dates=["date"])
    first = targets["date"].min() - pd.Timedelta(days=3)
    last = targets["date"].max() + pd.Timedelta(days=1)

    wind = build_wind(
        read_ndbc(a.wind_dir), first, last, a.cache,
        a.availability_lag_hours)
    wl = fetch_coops("water_level", first, last, a.cache)
    level = daily_aggregate(
        wl["Water Level"].astype(float), a.availability_lag_hours,
        "coops_8723214_wl_msl", "water_level_m")

    daily = wind.join(level, how="outer").join(
        read_rain(a.rain_dir, a.rain_availability_lag_hours), how="outer")
    daily = daily.loc[
        (daily.index >= first) & (daily.index <= last)].copy()

    # Keep separate field-level provenance. These are shared regional
    # measurements, not independent local observations at each target node.
    for field in ("rain_mm", "wind_mps", "water_level_m"):
        is_measured = daily[field].notna()
        daily[f"{field}_spatial_scope"] = np.where(
            is_measured, "regional_proxy", "")
        daily[f"{field}_coverage_basis"] = np.where(
            is_measured,
            ("daily_record_present_not_subdaily_verified"
             if field == "rain_mm" else "observed_six_minute_slots"),
            "")
        daily[f"{field}_availability_basis"] = np.where(
            is_measured, "assumed_fixed_lag_not_verified", "")
        daily[f"{field}_role"] = np.where(
            is_measured,
            ("coastal_tidal_boundary" if field == "water_level_m"
             else "regional_meteorology"), "")

    daily["source_id"] = (
        daily["wind_mps_source"].fillna("none") + "+" +
        daily["water_level_m_source"].fillna("none") + "+" +
        daily["rain_mm_source"].fillna("none"))
    cols = ["source_id"]
    for field in ("rain_mm", "wind_mps", "water_level_m"):
        for suffix in (
            "", "_observed_end_utc", "_available_at_utc", "_coverage",
            "_source", "_spatial_scope", "_coverage_basis",
            "_availability_basis", "_role",
        ):
            col = field + suffix
            if col not in daily:
                daily[col] = np.nan
            cols.append(col)
    daily = daily[cols].reset_index()
    nodes = sorted(targets["node"].unique())
    out = pd.concat(
        [daily.assign(node=node) for node in nodes], ignore_index=True)
    out = out[["node", "date"] + cols]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for col in out.columns:
        if col.endswith("_utc"):
            out[col] = pd.to_datetime(out[col], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(out)} rows, {len(nodes)} nodes, "
          f"{daily['date'].min():%Y-%m-%d}..{daily['date'].max():%Y-%m-%d}")
    print(f"wind days: {int(daily['wind_mps'].notna().sum())}, "
          f"water-level days: {int(daily['water_level_m'].notna().sum())}")
    print("NOTE: shared regional proxies; daily MIA rainfall coverage is "
          "record presence; all publication lags are assumptions.")


if __name__ == "__main__":
    main()
