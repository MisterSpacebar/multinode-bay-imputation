"""
map_viz.py
----------
Geographic bubble-map visualisations of imputed sensor data over North Biscayne Bay.

Produces four figures saved to visualizations/:
  25_map_seasonal_temp.png      — Water temperature, seasonal medians by year
  26_map_seasonal_sal.png       — Salinity, seasonal medians by year
  27_map_seasonal_do.png        — Dissolved oxygen (mg/L), seasonal medians by year
  28_map_seasonal_turb.png      — Turbidity (FNU), seasonal medians by year
  29_map_annual_overview.png    — Annual median for all four variables side-by-side

Run:
    python map_viz.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import contextily as ctx
from pathlib import Path

# ── pyproj is installed as a contextily dependency ──────────────────────────
from pyproj import Transformer

OUT_DIR     = Path("visualizations")
IMPUTED_DIR = Path("imputed_output")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------
STATIONS = {
    "L0":                  {"lat": 25.9114, "lon": -80.1373, "label": "L0 (Haulover)"},
    "L1":                  {"lat": 25.8749, "lon": -80.1832, "label": "L1 (B.Canal)"},
    "L2":                  {"lat": 25.8537, "lon": -80.1594, "label": "L2 (Little R.)"},
    "L6":                  {"lat": 25.8442, "lon": -80.1489, "label": "L6 (NBV N.)"},
    "L7":                  {"lat": 25.7791, "lon": -80.2089, "label": "L7 (Miami R.)"},
    "biscayne_bay":        {"lat": 25.8704, "lon": -80.1649, "label": "Bay (p2)"},
    "consolidated_crest5": {"lat": 25.7270, "lon": -80.2697, "label": "Crest5 (S.Bay)"},
}

IMPUTED_FILES = {
    "L0":                  "raw-data-platformL0_parameters_imputed.csv",
    "L1":                  "raw-data-platformL1_parameters_imputed.csv",
    "L2":                  "raw-data-platformL2_parameters_imputed.csv",
    "L6":                  "raw-data-platformL6_parameters_imputed.csv",
    "L7":                  "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay":        "biscayne_bay_imputed.csv",
    "consolidated_crest5": "consolidated_crest5_imputed.csv",
}

# Season definitions (meteorological)
SEASONS = {
    "Winter (DJF)": [12, 1, 2],
    "Spring (MAM)": [3, 4, 5],
    "Summer (JJA)": [6, 7, 8],
    "Autumn (SON)": [9, 10, 11],
}
SEASON_ORDER = list(SEASONS.keys())

VARIABLES = {
    "temp_c":        {"label": "Water Temp (°C)",      "cmap": "RdYlBu_r", "vmin": 18,  "vmax": 34},
    "sal_ppt":       {"label": "Salinity (PPT)",        "cmap": "YlOrBr",   "vmin": 20,  "vmax": 38},
    "odo_mgL":       {"label": "DO (mg/L)",             "cmap": "RdYlGn",   "vmin": 2,   "vmax": 10},
    "turbidity_fnu": {"label": "Turbidity (FNU)",       "cmap": "YlOrRd",   "vmin": 0,   "vmax": 15},
}

# Web Mercator transformer
_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_merc(lon, lat):
    return _to_mercator.transform(lon, lat)

# ---------------------------------------------------------------------------
# Load and aggregate data
# ---------------------------------------------------------------------------

def load_all_observed() -> dict[str, pd.DataFrame]:
    """Load imputed CSVs; keep only rows that were genuinely observed."""
    nodes = {}
    for name, fname in IMPUTED_FILES.items():
        path = IMPUTED_DIR / fname
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        # Use observed rows only for aggregate statistics
        obs_cols = [c for c in df.columns if c.endswith("_observed")]
        feat_cols = [c.replace("_observed", "") for c in obs_cols]
        for fc, oc in zip(feat_cols, obs_cols):
            df.loc[df[oc] == 0, fc] = np.nan   # mask imputed values
        nodes[name] = df[feat_cols]
    return nodes


def aggregate(nodes: dict, freq: str = "season") -> pd.DataFrame:
    """
    Returns a DataFrame with columns: station, year, season, variable, value.
    freq: 'season' or 'year'
    """
    records = []
    for name, df in nodes.items():
        df = df.copy()
        df.index = df.index.tz_convert(None)   # drop tz for grouping
        df["year"]   = df.index.year
        df["month"]  = df.index.month

        # Assign season label
        month_to_season = {}
        for sname, months in SEASONS.items():
            for m in months:
                month_to_season[m] = sname
        df["season"] = df["month"].map(month_to_season)

        # Winter belongs to the following year (Dec → next year's DJF)
        df.loc[df["month"] == 12, "year"] = df.loc[df["month"] == 12, "year"] + 1

        for var in VARIABLES:
            if var not in df.columns:
                continue
            if freq == "season":
                grp = df.groupby(["year", "season"])[var].median()
                for (yr, ss), val in grp.items():
                    records.append({"station": name, "year": yr,
                                    "season": ss, "variable": var, "value": val})
            else:  # annual
                grp = df.groupby("year")[var].median()
                for yr, val in grp.items():
                    records.append({"station": name, "year": yr,
                                    "season": "Annual", "variable": var, "value": val})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _basemap_bounds():
    """Mercator bounding box for North + South Biscayne Bay with margin."""
    lons = [s["lon"] for s in STATIONS.values()]
    lats = [s["lat"] for s in STATIONS.values()]
    pad_lon, pad_lat = 0.05, 0.04
    x0, y0 = to_merc(min(lons) - pad_lon, min(lats) - pad_lat)
    x1, y1 = to_merc(max(lons) + pad_lon, max(lats) + pad_lat)
    return x0, y0, x1, y1


def _draw_basemap(ax):
    x0, y0, x1, y1 = _basemap_bounds()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                        zoom=12, attribution=False)
    except Exception:
        ax.set_facecolor("#d4e9f7")


def _scatter_stations(ax, row_data: dict, var_cfg: dict, size=420):
    """
    Plot one bubble per station. row_data = {station_name: value or NaN}.
    Returns the scatter artist for colorbar use.
    """
    xs, ys, vals = [], [], []
    for name, meta in STATIONS.items():
        x, y = to_merc(meta["lon"], meta["lat"])
        xs.append(x); ys.append(y)
        vals.append(row_data.get(name, np.nan))

    norm = mcolors.Normalize(vmin=var_cfg["vmin"], vmax=var_cfg["vmax"])
    cmap = plt.get_cmap(var_cfg["cmap"])

    for x, y, v, name in zip(xs, ys, vals, STATIONS):
        color = cmap(norm(v)) if not np.isnan(v) else "lightgrey"
        ax.scatter(x, y, s=size, c=[color], edgecolors="white",
                   linewidths=1.2, zorder=5)
        label = STATIONS[name]["label"]
        ax.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points",
                    fontsize=6.5, fontweight="bold", color="#222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    # invisible scatter just to return for colorbar
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    return sc


# ---------------------------------------------------------------------------
# Figure 1–4: one figure per variable, rows=years, cols=seasons
# ---------------------------------------------------------------------------

def plot_seasonal_maps(agg_df: pd.DataFrame):
    years = sorted(agg_df["year"].unique())

    for var, cfg in VARIABLES.items():
        subset = agg_df[agg_df["variable"] == var]
        n_years   = len(years)
        n_seasons = len(SEASON_ORDER)

        fig, axes = plt.subplots(
            n_years, n_seasons,
            figsize=(n_seasons * 3.4, n_years * 3.6),
            subplot_kw={"projection": None},
        )
        if n_years == 1:
            axes = axes[np.newaxis, :]

        for ri, yr in enumerate(years):
            for ci, ss in enumerate(SEASON_ORDER):
                ax = axes[ri, ci]
                _draw_basemap(ax)

                cell = subset[(subset["year"] == yr) & (subset["season"] == ss)]
                row_data = dict(zip(cell["station"], cell["value"]))

                sc = _scatter_stations(ax, row_data, cfg)

                if ri == 0:
                    ax.set_title(ss, fontsize=8.5, fontweight="bold", pad=4)
                if ci == 0:
                    ax.set_ylabel(str(yr), fontsize=9, fontweight="bold", labelpad=4)

                ax.set_xticks([]); ax.set_yticks([])

        # shared colorbar
        norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
        sm   = plt.cm.ScalarMappable(cmap=plt.get_cmap(cfg["cmap"]), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                            fraction=0.02, pad=0.02, shrink=0.8)
        cbar.set_label(cfg["label"], fontsize=9)

        # grey patch legend for missing
        grey_patch = mpatches.Patch(facecolor="lightgrey", edgecolor="white",
                                    label="No observed data")
        fig.legend(handles=[grey_patch], loc="lower left",
                   fontsize=8, framealpha=0.7)

        fig.suptitle(f"North Biscayne Bay — {cfg['label']}\nSeasonal medians"
                     " (observed values only, grey = no data)",
                     fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()

        idx = list(VARIABLES).index(var) + 25
        out = OUT_DIR / f"{idx:02d}_map_seasonal_{var}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Figure 5: annual overview — 2×2 grid, one panel per variable
# ---------------------------------------------------------------------------

def plot_annual_overview(agg_df: pd.DataFrame):
    annual = agg_df[agg_df["season"] == "Annual"]
    years  = sorted(annual["year"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ai, (var, cfg) in enumerate(VARIABLES.items()):
        ax_outer = axes[ai]
        ax_outer.set_visible(False)   # used only for title space

        subset = annual[annual["variable"] == var]
        norm   = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
        cmap   = plt.get_cmap(cfg["cmap"])

        n_years = len(years)
        # create sub-axes inside each quadrant
        left   = ax_outer.get_position().x0
        bottom = ax_outer.get_position().y0
        width  = ax_outer.get_position().width
        height = ax_outer.get_position().height
        sub_w  = width / n_years - 0.005
        sub_h  = height - 0.04

        for yi, yr in enumerate(years):
            sub_ax = fig.add_axes([
                left + yi * (sub_w + 0.005),
                bottom + 0.035,
                sub_w,
                sub_h,
            ])
            _draw_basemap(sub_ax)
            cell     = subset[subset["year"] == yr]
            row_data = dict(zip(cell["station"], cell["value"]))
            _scatter_stations(sub_ax, row_data, cfg, size=300)
            sub_ax.set_title(str(yr), fontsize=8, pad=2)
            sub_ax.set_xticks([]); sub_ax.set_yticks([])

        # variable label
        fig.text(
            left + width / 2,
            bottom + 0.005,
            cfg["label"],
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

        # mini colorbar per panel
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        sub_cbar_ax = fig.add_axes([
            left + width + 0.003,
            bottom + 0.05,
            0.008,
            sub_h - 0.04,
        ])
        fig.colorbar(sm, cax=sub_cbar_ax)
        sub_cbar_ax.tick_params(labelsize=7)

    fig.suptitle("North Biscayne Bay — Annual Median (observed data only)\n"
                 "Bubble colour = measured value at each station",
                 fontsize=12, fontweight="bold", y=1.01)

    out = OUT_DIR / "29_map_annual_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading imputed data (observed values only)...")
    nodes = load_all_observed()

    print("Computing seasonal aggregates...")
    seasonal_agg = aggregate(nodes, freq="season")

    print("Computing annual aggregates...")
    annual_agg   = aggregate(nodes, freq="year")
    annual_agg["season"] = "Annual"

    combined = pd.concat([seasonal_agg, annual_agg], ignore_index=True)

    print("\nGenerating seasonal maps (one per variable)...")
    plot_seasonal_maps(seasonal_agg)

    print("\nGenerating annual overview map...")
    plot_annual_overview(annual_agg)

    print(f"\nDone — figures saved to {OUT_DIR}/")
