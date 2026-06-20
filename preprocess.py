"""
preprocess.py
-------------
Loads all water station CSVs and rainfall, aligns them on a common time grid,
builds a spatial graph based on haversine distance between stations, and
returns everything as tensors ready for the ST-GNN.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "water_data"
STATION_DIR = DATA_DIR / "water_stations_raw"
RAINFALL_CSV = DATA_DIR / "rainfall" / "rainfall_2025_mar_oct_combined.csv"

STATION_FILES = sorted(STATION_DIR.glob("raw-data-platform*.csv"))

# Sensor columns common to all stations (TSS present only in some — excluded
# here to keep a uniform feature tensor; add it back if you drop-fill).
SENSOR_COLS = [
    "Temperature (C)",
    "Specific Conductance (uS/cm)",
    "Salinity (PPT)",
    "Pressure (psia)",
    "Depth (m)",
    "ODO (%Sat)",
    "ODO (mg/L)",
    "Turbidity (FNU)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_edge_index_and_weights(coords, k=4, self_loops=False):
    """
    Build a k-NN spatial graph from station coordinates.

    Parameters
    ----------
    coords : list of (lat, lon) tuples, length = N_nodes
    k      : number of nearest neighbours per node
    self_loops : include self-edges (useful for some GNN formulations)

    Returns
    -------
    edge_index : (2, E) int64 numpy array  [source, target]
    edge_weight: (E,)  float32 — inverse-distance weight, normalised 0-1
    dist_matrix: (N, N) full pairwise distances in km
    """
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine_km(*coords[i], *coords[j])
            else:
                D[i, j] = 0.0

    src, dst, weights = [], [], []
    for i in range(n):
        # sort neighbours by distance (ascending), skip self
        neighbours = np.argsort(D[i])
        neighbours = [j for j in neighbours if j != i][:k]
        for j in neighbours:
            src.append(i)
            dst.append(j)
            weights.append(1.0 / (D[i, j] + 1e-6))

    if self_loops:
        for i in range(n):
            src.append(i)
            dst.append(i)
            weights.append(1.0)

    edge_index = np.array([src, dst], dtype=np.int64)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.max()           # normalise to [0, 1]
    return edge_index, weights, D


# ---------------------------------------------------------------------------
# Load stations
# ---------------------------------------------------------------------------

def load_station(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, low_memory=False)

    # Skip stations that have no coordinate columns
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print(f"  [SKIP] {path.name} — no lat/lon columns")
        return None

    # Timestamp is a Unix epoch float
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Take the median lat/lon as the fixed location for this station
    lat = df["latitude"].median()
    lon = df["longitude"].median()

    # Keep only needed columns
    available = [c for c in SENSOR_COLS if c in df.columns]
    df = df[["datetime"] + available].copy()
    df.attrs["lat"] = lat
    df.attrs["lon"] = lon
    df.attrs["name"] = path.stem
    return df


def resample_station(df: pd.DataFrame, freq="5min") -> pd.DataFrame:
    """Aggregate high-frequency readings to a regular grid, forward-fill short gaps."""
    df = df.set_index("datetime")
    # Aggregate (mean) to the target frequency — handles any sub-minute resolution
    df = df.resample(freq).mean()
    df = df.ffill(limit=3)    # fill gaps ≤ 3 steps (≤ 15 min by default)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Load rainfall
# ---------------------------------------------------------------------------

def load_rainfall() -> pd.Series:
    """Returns a daily rainfall series indexed by UTC date, in inches."""
    df = pd.read_csv(RAINFALL_CSV)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")["rain_in"].sort_index()
    return df


def align_rainfall(rain: pd.Series, time_index: pd.DatetimeIndex) -> np.ndarray:
    """Broadcast daily rainfall to the station time grid (5-min steps)."""
    # Strip timezone so the date lookup matches the tz-naive rainfall index
    dates = time_index.tz_convert("UTC").normalize().tz_localize(None)
    rain_reindexed = rain.reindex(dates, method="ffill").fillna(0.0).values.astype(np.float32)
    return rain_reindexed


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def build_dataset(freq="5min", k_neighbours=4):
    """
    Returns
    -------
    X          : (T, N, F)  float32 — sensor readings (NaN where missing)
    rain       : (T,)       float32 — aligned daily rainfall
    edge_index : (2, E)     int64
    edge_weight: (E,)       float32
    coords     : list of (lat, lon)
    node_names : list of str
    time_index : DatetimeIndex
    feature_names : list of str
    """
    print("Loading stations...")
    stations = [s for s in (load_station(p) for p in STATION_FILES) if s is not None]

    # Use the UNION of all station time ranges so no station is discarded.
    # Stations will have NaN outside their own deployment window — the GNN
    # treats those as missing and imputes from spatial neighbours.
    resampled_all = [resample_station(s, freq=freq) for s in stations]
    t_min = min(df.index.min() for df in resampled_all)
    t_max = max(df.index.max() for df in resampled_all)

    print(f"Union time window: {t_min.date()}  →  {t_max.date()}")
    common_index = pd.date_range(t_min, t_max, freq=freq, tz="UTC")
    T = len(common_index)

    coords = []
    node_names = []
    resampled = []

    for s, r in zip(stations, resampled_all):
        r = r.reindex(common_index)   # align to common grid (NaN outside deployment)
        coords.append((s.attrs["lat"], s.attrs["lon"]))
        node_names.append(s.attrs["name"])
        resampled.append(r)

    N = len(stations)

    # Build feature tensor — use intersection of available sensor columns
    all_cols = [set(r.columns) for r in resampled]
    common_cols = sorted(set(SENSOR_COLS).intersection(*all_cols))
    print(f"Common sensor features: {common_cols}")

    X = np.full((T, N, len(common_cols)), np.nan, dtype=np.float32)
    for i, r in enumerate(resampled):
        X[:, i, :] = r[common_cols].values.astype(np.float32)

    # Rainfall
    print("Loading rainfall...")
    rain_series = load_rainfall()
    rain = align_rainfall(rain_series, common_index)

    # Graph
    print("Building spatial graph...")
    edge_index, edge_weight, dist_matrix = build_edge_index_and_weights(coords, k=k_neighbours)
    print(f"Graph: {N} nodes, {edge_index.shape[1]} edges")
    print("Distance matrix (km):")
    for i, name in enumerate(node_names):
        row = "  ".join(f"{dist_matrix[i,j]:5.2f}" for j in range(N))
        print(f"  {name}: [{row}]")

    return {
        "X": X,                          # (T, N, F)
        "rain": rain,                    # (T,)
        "edge_index": edge_index,        # (2, E)
        "edge_weight": edge_weight,      # (E,)
        "coords": coords,
        "node_names": node_names,
        "time_index": common_index,
        "feature_names": common_cols,
        "dist_matrix": dist_matrix,
    }


if __name__ == "__main__":
    data = build_dataset()
    X = data["X"]
    print(f"\nTensor shape: X={X.shape}  (timesteps × stations × features)")
    nan_pct = np.isnan(X).mean() * 100
    print(f"Overall missing: {nan_pct:.1f}%")
    for i, name in enumerate(data["node_names"]):
        n = np.isnan(X[:, i, :]).mean() * 100
        print(f"  {name}: {n:.1f}% missing")
