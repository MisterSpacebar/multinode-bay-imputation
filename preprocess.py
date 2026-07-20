"""
preprocess.py
-------------
Multi-year, multi-schema loader for bay water station data.

  2025 data : water_data/water_stations_raw/
              8 sensor channels, Unix float timestamps, lat/lon per row
  2026 data : water_data/water_stations_2026/
              3 sensor channels (temp, sal, ODO), local Miami time strings,
              GPS lat/lon per row

Both datasets are harmonised to a single canonical 8-feature set.  Stations
from both years are matched by haversine distance; co-located 2025 deployments
(L2, L3, L5 are all at the same spot) are merged via nanmean before matching.
Unmatched 2026 platforms become new graph nodes.

Rainfall is available for 2025 (Mar-Oct) and 2026 (Jan-May) with measured
rain_in, temp_min, and temp_max from Miami Airport daily records. All three
are broadcast to the 5-min grid and fed to the model as external forcing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = Path(__file__).parent / "water_data"
STATION_DIR   = DATA_DIR / "water_stations_raw"
STATION_2026  = DATA_DIR / "water_stations_2026"
RAINFALL_DIR  = DATA_DIR / "rainfall"

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "Temperature (C)":               "temp_c",
    "Specific Conductance (uS/cm)":  "spec_cond_uScm",
    "Salinity (PPT)":                "sal_ppt",
    "Pressure (psia)":               "pressure_psia",
    "Depth (m)":                     "depth_m",
    "ODO (%Sat)":                    "odo_pct",
    "ODO (mg/L)":                    "odo_mgL",
    "Turbidity (FNU)":               "turbidity_fnu",
    "exo.temp_C":                    "temp_c",
    "exo.sal_ppt":                   "sal_ppt",
    "exo.odo_mgL":                   "odo_mgL",
}

ALL_FEATURES = [
    "temp_c", "sal_ppt", "odo_mgL", "depth_m",
    "pressure_psia", "odo_pct", "spec_cond_uScm", "turbidity_fnu",
]

SAME_LOCATION_KM = 0.10

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlam / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_edge_index_and_weights(coords, k=4):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine_km(*coords[i], *coords[j])
    src, dst, weights = [], [], []
    for i in range(n):
        neighbours = [j for j in np.argsort(D[i]) if j != i][:k]
        for j in neighbours:
            src.append(i); dst.append(j)
            weights.append(1.0 / (D[i, j] + 1e-6))
    edge_index = np.array([src, dst], dtype=np.int64)
    w = np.array(weights, dtype=np.float32)
    w = w / w.max()
    return edge_index, w, D

# ---------------------------------------------------------------------------
# 2025 loader
# ---------------------------------------------------------------------------

def _load_raw_2025(path):
    df = pd.read_csv(path, low_memory=False)
    if "latitude" not in df.columns:
        print(f"  [SKIP] {path.name} - no lat/lon")
        return None
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("datetime")
    lat = float(df["latitude"].median())
    lon = float(df["longitude"].median())
    rename = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    df = df.rename(columns=rename)[["datetime"] + list(rename.values())].copy()
    df = df.set_index("datetime")
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df.attrs["lat"]  = lat
    df.attrs["lon"]  = lon
    df.attrs["name"] = path.stem
    return df[ALL_FEATURES]


def load_2025_stations():
    stations = []
    for p in sorted(STATION_DIR.glob("raw-data-platform*.csv")):
        df = _load_raw_2025(p)
        if df is not None:
            stations.append({"df": df, "lat": df.attrs["lat"],
                              "lon": df.attrs["lon"], "name": df.attrs["name"]})
    return stations

# ---------------------------------------------------------------------------
# Cluster co-located 2025 stations
# ---------------------------------------------------------------------------

def cluster_2025_stations(stations):
    n = len(stations)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = haversine_km(stations[i]["lat"], stations[i]["lon"],
                                    stations[j]["lat"], stations[j]["lon"])
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        group = [i]
        for j in range(i + 1, n):
            if not visited[j] and D[i, j] < SAME_LOCATION_KM:
                group.append(j)
                visited[j] = True
        members = [stations[k] for k in group]
        names   = [m["name"] for m in members]
        lat_c   = float(np.mean([m["lat"] for m in members]))
        lon_c   = float(np.mean([m["lon"] for m in members]))
        if len(members) == 1:
            df = members[0]["df"]
            name = names[0]
        else:
            print(f"  [MERGE] {names}")
            resampled = [m["df"].resample("5min").mean() for m in members]
            df = pd.concat(resampled).groupby(level=0).mean()
            name = names[0]   # use first name as canonical
        clusters.append({"df": df, "lat": lat_c, "lon": lon_c,
                          "name": name, "members": names})
    return clusters

# ---------------------------------------------------------------------------
# 2026 loader
# ---------------------------------------------------------------------------
MIAMI_TZ = "America/New_York"

def load_2026_station_group(folder):
    files = sorted(folder.glob("*.csv"))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["datetime"] = (
        pd.to_datetime(df["ts_local"])
          .dt.tz_localize(MIAMI_TZ, ambiguous="infer", nonexistent="shift_forward")
          .dt.tz_convert("UTC")
    )
    df = df.sort_values("datetime")
    df = df[df["sled.state"] == "at_bottom"].copy()
    lat = float(df["gps.lat"].median())
    lon = float(df["gps.lon"].median())
    pid = df["platform_id"].iloc[0]
    rename = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    df = df.rename(columns=rename)[["datetime"] + list(rename.values())].copy()
    df = df.set_index("datetime")
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    return {"df": df[ALL_FEATURES], "lat": lat, "lon": lon,
            "name": folder.name, "platform_id": pid}


def load_2026_stations():
    return [load_2026_station_group(d)
            for d in sorted(STATION_2026.iterdir()) if d.is_dir()]

# ---------------------------------------------------------------------------
# Merge 2026 into clusters
# ---------------------------------------------------------------------------

def merge_2026_into_clusters(clusters, stations_2026):
    for s26 in stations_2026:
        best_idx, best_d = None, float("inf")
        for idx, cl in enumerate(clusters):
            d = haversine_km(s26["lat"], s26["lon"], cl["lat"], cl["lon"])
            if d < best_d:
                best_d, best_idx = d, idx
        if best_d < SAME_LOCATION_KM:
            cl = clusters[best_idx]
            print(f"  [MATCH] 2026 {s26['name']} ({s26['platform_id']}) "
                  f"-> '{cl['name']}' ({best_d*1000:.0f} m)")
            combined = pd.concat([cl["df"], s26["df"]]).groupby(level=0).mean()
            clusters[best_idx] = {**cl, "df": combined}
        else:
            print(f"  [NEW NODE] 2026 {s26['name']} ({s26['platform_id']}) "
                  f"(nearest={best_d:.2f} km)")
            clusters.append({"df": s26["df"], "lat": s26["lat"], "lon": s26["lon"],
                              "name": s26["name"], "members": [s26["name"]]})
    return clusters

# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

def resample_to_grid(df, freq="5min"):
    df = df.resample(freq).mean()
    df = df.ffill(limit=3)
    return df

# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------

def load_weather() -> pd.DataFrame:
    """
    Load all available daily weather records (2025 + 2026) into a single
    DataFrame indexed by tz-naive UTC date, columns: rain_in, temp_min, temp_max.
    Missing dates are filled with DOY climatology derived from observed data.
    """
    files = sorted(RAINFALL_DIR.glob("*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        dfs.append(df[["date", "rain_in", "temp_min", "temp_max"]])
    weather = pd.concat(dfs, ignore_index=True)
    weather = weather.drop_duplicates("date").set_index("date").sort_index()
    return weather


def _build_forcing_array(weather: pd.DataFrame, col: str,
                         time_index: pd.DatetimeIndex) -> np.ndarray:
    """Broadcast a daily weather column to the 5-min time_index.
    Dates outside the measured range are filled with DOY climatology."""
    series = weather[col]
    doy_clim = series.groupby(series.index.dayofyear).mean()
    dates = time_index.tz_convert("UTC").normalize().tz_localize(None)
    result = np.zeros(len(dates), dtype=np.float32)
    idx_set = set(series.index)
    for i, d in enumerate(dates):
        if d in idx_set:
            result[i] = series[d]
        else:
            result[i] = float(doy_clim.get(d.dayofyear, 0.0))
    return result


def build_forcing_for_index(time_index: pd.DatetimeIndex):
    """
    Returns three (T,) float32 arrays aligned to time_index:
        rain     — daily precipitation (inches)
        temp_min — daily minimum air temperature (°C)
        temp_max — daily maximum air temperature (°C)
    """
    weather = load_weather()
    rain     = _build_forcing_array(weather, "rain_in",  time_index)
    temp_min = _build_forcing_array(weather, "temp_min", time_index)
    temp_max = _build_forcing_array(weather, "temp_max", time_index)
    return rain, temp_min, temp_max

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(freq="5min", k_neighbours=4):
    print("Loading 2025 stations...")
    raw_2025 = load_2025_stations()
    clusters = cluster_2025_stations(raw_2025)

    print("\nLoading 2026 stations...")
    raw_2026 = load_2026_stations()
    clusters = merge_2026_into_clusters(clusters, raw_2026)

    print(f"\n{len(clusters)} nodes total. Resampling...")
    resampled = [resample_to_grid(cl["df"], freq) for cl in clusters]

    t_min = min(df.index.min() for df in resampled)
    t_max = max(df.index.max() for df in resampled)
    common_index = pd.date_range(t_min, t_max, freq=freq, tz="UTC")
    T = len(common_index)

    resampled = [df.reindex(common_index) for df in resampled]

    N = len(clusters)
    F = len(ALL_FEATURES)
    X = np.full((T, N, F), np.nan, dtype=np.float32)
    for i, df in enumerate(resampled):
        for j, feat in enumerate(ALL_FEATURES):
            if feat in df.columns:
                X[:, i, j] = df[feat].values.astype(np.float32)

    print("Building forcing (rain + air temp)...")
    rain, temp_min, temp_max = build_forcing_for_index(common_index)
    print(f"  rain range:     {rain.min():.2f} – {rain.max():.2f} in")
    print(f"  air temp range: {temp_min.min():.1f} – {temp_max.max():.1f} °C")

    print("Building spatial graph...")
    coords = [(cl["lat"], cl["lon"]) for cl in clusters]
    edge_index, edge_weight, dist_matrix = build_edge_index_and_weights(coords, k=k_neighbours)

    node_names  = [cl["name"] for cl in clusters]
    short_names = [n.replace("raw-data-platformL", "L").replace("_parameters", "")
                   for n in node_names]

    print(f"\nGraph: {N} nodes, {edge_index.shape[1]} edges")
    print(f"Time: {t_min.date()} -> {t_max.date()}  ({T} steps)")
    print(f"Tensor: {T} x {N} x {F}")

    nan_pct = np.isnan(X).mean() * 100
    print(f"Overall missing: {nan_pct:.1f}%")
    for i, sn in enumerate(short_names):
        pct = np.isnan(X[:, i, :]).mean() * 100
        print(f"  {sn:40s}: {pct:.1f}% missing")

    return {
        "X":             X,
        "rain":          rain,
        "temp_min":      temp_min,
        "temp_max":      temp_max,
        "edge_index":    edge_index,
        "edge_weight":   edge_weight,
        "coords":        coords,
        "node_names":    node_names,
        "time_index":    common_index,
        "feature_names": ALL_FEATURES,
        "dist_matrix":   dist_matrix,
    }


# ---------------------------------------------------------------------------
# 2021-2024 historical grab-sample loader (for FCM nutrient analysis)
# ---------------------------------------------------------------------------

GRAB_SAMPLE_CSV  = DATA_DIR / "2021_to_2024" / "BB_thru2024.csv"

# Columns to keep from the grab-sample file, mapped to canonical names
GRAB_RENAME = {
    "temp":        "temp_c",
    "sal":         "sal_ppt",
    "spc_scm":     "spec_cond_uScm",
    "do_per":      "odo_pct",
    "do_mgL":      "odo_mgL",
    "ph":          "ph",
    "chl_exo_ugL": "chl_a_ugL",
    "secchi":      "secchi_m",
    "no2no3":      "no2no3_umolL",
    "nh4":         "nh4_umolL",
    "po4":         "po4_umolL",
    "din":         "din_umolL",
}

# Nutrient / extra features that appear only in the grab-sample data
GRAB_EXTRA_FEATURES = ["ph", "chl_a_ugL", "secchi_m", "no2no3_umolL",
                        "nh4_umolL", "po4_umolL", "din_umolL"]


def load_historical_grab_samples(
    site_types: list[str] | None = None,
    sample_type: str = "Surface",
) -> pd.DataFrame:
    """
    Load the 2021-2024 discrete grab-sample data and return a monthly
    DataFrame spatially averaged across all qualifying sites.

    Parameters
    ----------
    site_types  : list of site_type strings to include, e.g.
                  ['Biscayne Bay'].  None = all types.
    sample_type : 'Surface' | 'Bottom' | None (None = both).

    Returns
    -------
    monthly_df : DatetimeIndex (month-start, UTC-naive), columns = canonical
                 feature names (overlapping + extra nutrient features).
                 Missing months are NaN.
    """
    df = pd.read_csv(GRAB_SAMPLE_CSV, encoding="latin-1", low_memory=False)

    # Parse datetime (EST → treat as tz-naive; close enough for monthly agg)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])

    # Filters
    if sample_type is not None:
        df = df[df["sample_type"] == sample_type]
    if site_types is not None:
        df = df[df["site_type"].isin(site_types)]

    # Keep and rename relevant columns
    keep = [c for c in GRAB_RENAME if c in df.columns]
    sub = df[["datetime"] + keep].copy()
    sub = sub.rename(columns=GRAB_RENAME)
    sub = sub.set_index("datetime")

    # Convert all columns to numeric
    for col in sub.columns:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    # Monthly spatial average across all included sites
    monthly = sub.resample("MS").mean()  # MS = month-start

    return monthly


if __name__ == "__main__":
    build_dataset()
