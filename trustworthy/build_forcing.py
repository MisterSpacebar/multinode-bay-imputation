"""Build the daily forcing CSV consumed by time_valid_forcing / onset_forcing.

Sources (Virginia Key, NOAA NOS 8723214 / NDBC VAKF1):
  * wind:  local NDBC annual files (dates <= last NDBC day), NOAA CO-OPS API after
  * water level: NOAA CO-OPS API, MSL datum, metric, GMT
Rain is the Miami airport daily total (inches -> mm) from water_data/rainfall; its
availability time is an assumption (--rain-availability-lag-hours after local-day end).
Virginia Key is not a station-local sensor. Its daily values are written to every
station that has daily targets, as an explicit analyst election, and the source_id
records the origin. available_at_utc is observation end plus a stated assumed
publication lag (--availability-lag-hours); it is NOT a verified timestamp.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/New_York"
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION = "8723214"
SLOTS_PER_HOUR = 10  # 6-minute data


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
            with urllib.request.urlopen(url, timeout=120) as r:
                text = r.read().decode("utf-8")
            if text.lstrip().startswith("Error") or "Date Time" not in text:
                raise RuntimeError(f"CO-OPS {product} {cur:%Y-%m-%d}: {text[:200]}")
            path.write_text(text, encoding="utf-8")
        frames.append(pd.read_csv(path, skipinitialspace=True))
        cur = stop + pd.Timedelta(days=1)
    d = pd.concat(frames, ignore_index=True)
    d.columns = [c.strip() for c in d.columns]
    d["time"] = pd.to_datetime(d["Date Time"], utc=True)
    return d.drop_duplicates("time").set_index("time").sort_index()


def read_ndbc(folder: Path) -> pd.Series:
    parts = []
    for p in sorted(folder.glob("vakf1h*.txt")):
        d = pd.read_csv(p, sep=r"\s+", comment="#", header=None, usecols=range(7),
                        names=["yy", "mo", "dy", "hh", "mm", "wdir", "wspd"])
        t = pd.to_datetime(dict(year=d.yy, month=d.mo, day=d.dy, hour=d.hh, minute=d.mm),
                           utc=True)
        s = pd.Series(d.wspd.to_numpy(float), index=t)
        parts.append(s.where(s < 90))  # 99.0 is the NDBC missing flag
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index()


def daily_aggregate(series: pd.Series, lag_hours: float, source_id: str,
                    name: str) -> pd.DataFrame:
    s = series.dropna()
    local_date = s.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    g = s.groupby(local_date)
    out = pd.DataFrame({name: g.mean(), "n": g.size(), "last": g.apply(lambda x: x.index.max())})
    day_start = out.index.tz_localize(LOCAL_TZ).tz_convert("UTC")
    day_end = (out.index + pd.DateOffset(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    hours = (day_end - day_start).total_seconds() / 3600
    out[f"{name}_coverage"] = (out["n"] / (hours * SLOTS_PER_HOUR)).clip(upper=1.0)
    end = (out["last"] + pd.Timedelta(minutes=6)).clip(upper=pd.Series(day_end, index=out.index))
    out[f"{name}_observed_end_utc"] = end
    out[f"{name}_available_at_utc"] = end + pd.Timedelta(hours=lag_hours)
    out[f"{name}_source"] = source_id
    out.index.name = "date"
    return out.drop(columns=["n", "last"])


def read_rain(folder: Path, lag_hours: float) -> pd.DataFrame:
    """Miami airport daily totals (inches, local day); availability is assumed."""
    files = [folder / "rainfall_2025_mar_oct_combined.csv",
             *sorted(folder.glob("miami_airport_daily_2026_*.csv"))]
    d = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))
    d = d.drop_duplicates("date").set_index("date").sort_index()
    out = pd.DataFrame({"rain_mm": pd.to_numeric(d["rain_in"], errors="coerce") * 25.4})
    out = out.dropna()
    end = (out.index + pd.DateOffset(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    out["rain_mm_coverage"] = 1.0
    out["rain_mm_observed_end_utc"] = end
    out["rain_mm_available_at_utc"] = end + pd.Timedelta(hours=lag_hours)
    out["rain_mm_source"] = "mia_airport_daily"
    out.index.name = "date"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rain-dir", type=Path, default=Path("water_data/rainfall"))
    p.add_argument("--rain-availability-lag-hours", type=float, default=24.0)
    p.add_argument("--daily", type=Path, default=Path("trustworthy_outputs/daily_targets.csv"))
    p.add_argument("--wind-dir", type=Path, default=Path("water_data/wind_data"))
    p.add_argument("--cache", type=Path, default=Path("water_data/coops_8723214"))
    p.add_argument("--out", type=Path, default=Path("water_data/forcing_daily.csv"))
    p.add_argument("--availability-lag-hours", type=float, default=3.0)
    a = p.parse_args()

    targets = pd.read_csv(a.daily, parse_dates=["date"])
    first = targets["date"].min() - pd.Timedelta(days=3)
    last = targets["date"].max() + pd.Timedelta(days=1)

    ndbc = read_ndbc(a.wind_dir)
    ndbc_end = ndbc.index.max()
    wind_ndbc = ndbc[ndbc.index <= ndbc_end]
    coops_start = (ndbc_end + pd.Timedelta(days=1)).tz_convert(None).normalize()
    wind_parts = [daily_aggregate(wind_ndbc, a.availability_lag_hours, "ndbc_vakf1", "wind_mps")]
    if coops_start <= last:
        w = fetch_coops("wind", coops_start, last, a.cache)["Speed"].astype(float)
        wind_parts.append(daily_aggregate(w, a.availability_lag_hours, "coops_8723214_wind",
                                          "wind_mps"))
    wind = pd.concat(wind_parts)
    wind = wind[~wind.index.duplicated()]

    wl = fetch_coops("water_level", first, last, a.cache)["Water Level"].astype(float)
    level = daily_aggregate(wl, a.availability_lag_hours, "coops_8723214_wl", "water_level_m")

    daily = wind.join(level, how="outer", lsuffix="_w", rsuffix="_l")
    daily = daily.join(read_rain(a.rain_dir, a.rain_availability_lag_hours), how="outer")
    daily = daily.loc[(daily.index >= first) & (daily.index <= last)]
    daily["source_id"] = (daily["wind_mps_source"].fillna("none") + "+"
                          + daily["water_level_m_source"].fillna("none") + "+"
                          + daily["rain_mm_source"].fillna("none"))
    cols = ["source_id"]
    for f in ("rain_mm", "wind_mps", "water_level_m"):
        for suffix in ("", "_observed_end_utc", "_available_at_utc", "_coverage"):
            c = f + suffix
            if c not in daily:
                daily[c] = np.nan
            cols.append(c)
    daily = daily[cols].reset_index()
    nodes = sorted(targets["node"].unique())
    out = pd.concat([daily.assign(node=n) for n in nodes], ignore_index=True)
    out = out[["node", "date"] + cols]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for c in out.columns:
        if c.endswith("_utc"):
            out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(out)} rows, {len(nodes)} nodes, "
          f"{daily['date'].min():%Y-%m-%d}..{daily['date'].max():%Y-%m-%d}")
    print(f"wind days: {int(daily['wind_mps'].notna().sum())}, "
          f"water-level days: {int(daily['water_level_m'].notna().sum())}")


if __name__ == "__main__":
    main()
