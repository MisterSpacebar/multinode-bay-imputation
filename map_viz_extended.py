"""
map_viz_extended.py
-------------------
Extended geographic maps combining 2021-2024 grab-sample sites with
2025-2026 continuous sensor stations across Biscayne Bay.

Adds figures 30-36 to visualizations/:
  30_map_ext_temp.png       — Water temp 2021-2026, all sites seasonal
  31_map_ext_sal.png        — Salinity 2021-2026, all sites seasonal
  32_map_ext_do.png         — DO (mg/L) 2021-2026, all sites seasonal
  33_map_nutrient_ph.png    — pH 2021-2024 (grab sites), seasonal
  34_map_nutrient_chl.png   — Chlorophyll-a 2021-2024, seasonal
  35_map_nutrient_din.png   — DIN 2021-2024, seasonal
  36_map_canal_stress.png   — Annual median NH4 + DO by site type (canal vs. bay)

Run:
    python map_viz_extended.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as ctx
from pathlib import Path
from pyproj import Transformer

OUT_DIR     = Path("visualizations")
IMPUTED_DIR = Path("imputed_output")
GRAB_CSV    = Path("water_data/2021_to_2024/BB_thru2024.csv")
OUT_DIR.mkdir(exist_ok=True)

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_merc(lon, lat):
    return _to_merc.transform(lon, lat)

# ---------------------------------------------------------------------------
# Sensor station metadata (2025-2026)
# ---------------------------------------------------------------------------
SENSOR_STATIONS = {
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

SEASONS = {
    "Winter (DJF)": [12, 1, 2],
    "Spring (MAM)": [3, 4, 5],
    "Summer (JJA)": [6, 7, 8],
    "Autumn (SON)": [9, 10, 11],
}
SEASON_ORDER = list(SEASONS.keys())

# Site-type colour for the canal-stress map
SITE_TYPE_COLOR = {
    "Biscayne Bay":       "#2196F3",
    "Little River Canal": "#F44336",
    "Miami Canal":        "#E91E63",
    "Inlet":              "#FF9800",
    "Outfall":            "#9C27B0",
    "Reef":               "#4CAF50",
}

# ---------------------------------------------------------------------------
# Load grab data
# ---------------------------------------------------------------------------

def load_grabs() -> pd.DataFrame:
    df = pd.read_csv(GRAB_CSV, encoding="latin-1")
    df = df[df["sample_type"] == "Surface"].copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Deduplicate site names: GOC001 → GOC-001, GOC014 → GOC-014 etc.
    df["site_name"] = df["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)

    # Season + winter year correction
    m2s = {}
    for sname, months in SEASONS.items():
        for m in months:
            m2s[m] = sname
    df["season"] = df["month"].map(m2s)
    df.loc[df["month"] == 12, "year"] = df.loc[df["month"] == 12, "year"] + 1

    # Canonical coords per site (median of all rows)
    site_coords = df.groupby("site_name")[["lat_dec", "lon_dec"]].median()
    df["lat"] = df["site_name"].map(site_coords["lat_dec"])
    df["lon"] = df["site_name"].map(site_coords["lon_dec"])
    df["site_type"] = df.groupby("site_name")["site_type"].transform("first")

    return df


def agg_grabs(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Median per (site, year, season)."""
    site_meta = df.groupby("site_name")[["lat", "lon", "site_type"]].first()
    vals = pd.to_numeric(df[var], errors="coerce")
    agg = (
        df.assign(**{var: vals})
          .groupby(["site_name", "year", "season"])[var]
          .median()
          .reset_index()
          .rename(columns={var: "value"})
    )
    agg = agg.join(site_meta, on="site_name")
    return agg


# ---------------------------------------------------------------------------
# Load sensor data (observed values only, seasonal medians)
# ---------------------------------------------------------------------------

def agg_sensors(feat: str) -> pd.DataFrame:
    records = []
    for name, fname in IMPUTED_FILES.items():
        path = IMPUTED_DIR / fname
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

        if feat not in df.columns:
            continue
        obs_col = feat + "_observed"
        if obs_col in df.columns:
            df.loc[df[obs_col] == 0, feat] = np.nan   # mask imputed

        df["year"]  = df.index.year
        df["month"] = df.index.month
        m2s = {}
        for sname, months in SEASONS.items():
            for m in months:
                m2s[m] = sname
        df["season"] = df["month"].map(m2s)
        df.loc[df["month"] == 12, "year"] = df.loc[df["month"] == 12, "year"] + 1

        grp = df.groupby(["year", "season"])[feat].median()
        meta = SENSOR_STATIONS[name]
        for (yr, ss), val in grp.items():
            records.append({"site_name": name, "lat": meta["lat"], "lon": meta["lon"],
                             "site_type": "Sensor", "year": yr, "season": ss, "value": val})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _bounds(extra_lons=None, extra_lats=None, pad_lon=0.06, pad_lat=0.05):
    lons = [s["lon"] for s in SENSOR_STATIONS.values()]
    lats = [s["lat"] for s in SENSOR_STATIONS.values()]
    if extra_lons:
        lons += list(extra_lons)
    if extra_lats:
        lats += list(extra_lats)
    x0, y0 = to_merc(min(lons) - pad_lon, min(lats) - pad_lat)
    x1, y1 = to_merc(max(lons) + pad_lon, max(lats) + pad_lat)
    return x0, y0, x1, y1


def _add_basemap(ax, bounds):
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                        zoom=11, attribution=False)
    except Exception:
        ax.set_facecolor("#d4e9f7")


def _plot_bubbles(ax, rows: pd.DataFrame, cfg: dict,
                  marker="o", size=220, edgewidth=1.2):
    """Scatter one bubble per row using (lat, lon, value)."""
    norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
    cmap = plt.get_cmap(cfg["cmap"])
    for _, r in rows.iterrows():
        x, y = to_merc(r["lon"], r["lat"])
        color = cmap(norm(r["value"])) if not np.isnan(r["value"]) else "lightgrey"
        ax.scatter(x, y, s=size, c=[color], marker=marker,
                   edgecolors="white", linewidths=edgewidth, zorder=5)
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    return sc


# ---------------------------------------------------------------------------
# Figures 30-32: combined 2021-2026  (years as columns, Summer/Winter as rows)
# ---------------------------------------------------------------------------

COMBINED_VARS = {
    "temp":   ("temp_c",  {"label": "Water Temp (°C)", "cmap": "RdYlBu_r", "vmin": 18, "vmax": 34}),
    "sal":    ("sal_ppt", {"label": "Salinity (PPT)",   "cmap": "YlOrBr",   "vmin": 10, "vmax": 38}),
    "do_mgL": ("odo_mgL", {"label": "DO (mg/L)",        "cmap": "RdYlGn",   "vmin": 2,  "vmax": 10}),
}

# Only the two most contrasting seasons to keep panels large and readable
KEY_SEASONS = ["Summer (JJA)", "Winter (DJF)"]

# Sites where we annotate the numeric value directly on the bubble
LABEL_SITES = {"LR01", "MR01", "GOC-014", "BB14", "BB25"}


def _annotate_values(ax, rows: pd.DataFrame, cfg: dict, fontsize=6.5):
    """Print rounded value next to each labelled grab site."""
    norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
    for _, r in rows.iterrows():
        if r["site_name"] not in LABEL_SITES or np.isnan(r["value"]):
            continue
        x, y = to_merc(r["lon"], r["lat"])
        txt  = f"{r['value']:.1f}"
        # dark text on light bubble, light text on dark bubble
        brightness = sum(plt.get_cmap(cfg["cmap"])(norm(r["value"]))[:3]) / 3
        fc = "white" if brightness < 0.55 else "#222"
        ax.annotate(txt, (x, y), ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=fc, zorder=7)


def plot_combined_seasonal(grabs: pd.DataFrame, fig_start=30):
    for fig_idx, (grab_col, (sensor_col, cfg)) in enumerate(COMBINED_VARS.items()):
        grab_agg   = agg_grabs(grabs, grab_col)
        sensor_agg = agg_sensors(sensor_col)

        all_lons = list(grab_agg["lon"]) + [s["lon"] for s in SENSOR_STATIONS.values()]
        all_lats = list(grab_agg["lat"]) + [s["lat"] for s in SENSOR_STATIONS.values()]
        bounds = _bounds(all_lons, all_lats)

        years = sorted(set(grab_agg["year"].unique()) | set(sensor_agg["year"].unique()))
        years = [y for y in years if 2021 <= y <= 2026]
        n_years = len(years)

        # 2 rows (seasons) × N columns (years) — landscape, panels are larger
        fig, axes = plt.subplots(
            2, n_years,
            figsize=(n_years * 3.6, 2 * 4.4),
            constrained_layout=False,
        )
        fig.subplots_adjust(wspace=0.04, hspace=0.12,
                            left=0.04, right=0.88, top=0.90, bottom=0.04)

        norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
        cmap = plt.get_cmap(cfg["cmap"])

        for ci, yr in enumerate(years):
            for ri, ss in enumerate(KEY_SEASONS):
                ax = axes[ri, ci]
                _add_basemap(ax, bounds)

                # Grab sites — circles
                g = grab_agg[(grab_agg["year"] == yr) & (grab_agg["season"] == ss)]
                if not g.empty:
                    _plot_bubbles(ax, g, cfg, marker="o", size=260, edgewidth=1.0)
                    _annotate_values(ax, g, cfg)

                # Sensor stations — diamonds (larger, on top)
                s = sensor_agg[(sensor_agg["year"] == yr) & (sensor_agg["season"] == ss)]
                if not s.empty:
                    _plot_bubbles(ax, s, cfg, marker="D", size=360, edgewidth=2.0)

                # Column header (year) on top row only
                if ri == 0:
                    ax.set_title(str(yr), fontsize=11, fontweight="bold", pad=5)
                # Row label (season) on leftmost column only
                if ci == 0:
                    ax.set_ylabel(ss.replace(" (", "\n("), fontsize=9,
                                  fontweight="bold", labelpad=4)
                ax.set_xticks([]); ax.set_yticks([])

                # Grey overlay when no data at all
                if g.empty and s.empty:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="#888")

        # Single shared colorbar on the right
        cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.78])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(cfg["label"], fontsize=10, labelpad=8)
        cb.ax.tick_params(labelsize=8)

        # Legend for marker shapes
        grab_h   = mlines.Line2D([], [], color="#777", marker="o", linestyle="",
                                  markersize=8,  label="Grab sample (2021–2024)")
        sensor_h = mlines.Line2D([], [], color="#777", marker="D", linestyle="",
                                  markersize=9,  label="Sensor station (2025–2026)")
        grey_h   = mlines.Line2D([], [], color="lightgrey", marker="o", linestyle="",
                                  markersize=8,  label="No observed data")
        fig.legend(handles=[grab_h, sensor_h, grey_h], loc="lower center",
                   ncol=3, fontsize=8.5, framealpha=0.85,
                   bbox_to_anchor=(0.45, -0.02))

        fig.suptitle(
            f"Biscayne Bay — {cfg['label']} · 2021–2026\n"
            "Columns = year  |  Rows = season  |  ◯ grab samples  ◆ sensors",
            fontsize=12, fontweight="bold",
        )

        out = OUT_DIR / f"{fig_start + fig_idx:02d}_map_ext_{grab_col}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Figures 33-35: nutrient maps (years as columns, Summer/Winter as rows)
# ---------------------------------------------------------------------------

NUTRIENT_VARS = {
    "ph":          {"label": "pH",           "cmap": "RdYlGn", "vmin": 7.4, "vmax": 8.5},
    "chl_exo_ugL": {"label": "Chl-a (µg/L)", "cmap": "YlGn",   "vmin": 0,   "vmax": 10},
    "din":         {"label": "DIN (µmol/L)",  "cmap": "YlOrRd", "vmin": 0,   "vmax": 20},
}


def plot_nutrient_seasonal(grabs: pd.DataFrame, fig_start=33):
    all_lons = list(grabs["lon"].dropna())
    all_lats = list(grabs["lat"].dropna())
    bounds   = _bounds(all_lons, all_lats, pad_lon=0.07, pad_lat=0.05)

    # Only grab years (2021-2024; Dec-corrected 2025 excluded)
    years = [y for y in sorted(grabs["year"].unique()) if y <= 2024]
    n_years = len(years)

    for fig_idx, (grab_col, cfg) in enumerate(NUTRIENT_VARS.items()):
        g_agg = agg_grabs(grabs, grab_col)

        # 2 rows (Summer / Winter) × N columns (years)
        fig, axes = plt.subplots(
            2, n_years,
            figsize=(n_years * 3.8, 2 * 4.6),
            constrained_layout=False,
        )
        fig.subplots_adjust(wspace=0.04, hspace=0.12,
                            left=0.06, right=0.88, top=0.90, bottom=0.04)

        norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
        cmap = plt.get_cmap(cfg["cmap"])

        for ci, yr in enumerate(years):
            for ri, ss in enumerate(KEY_SEASONS):
                ax = axes[ri, ci]
                _add_basemap(ax, bounds)

                g = g_agg[(g_agg["year"] == yr) & (g_agg["season"] == ss)]
                if not g.empty:
                    _plot_bubbles(ax, g, cfg, marker="o", size=310, edgewidth=1.2)
                    _annotate_values(ax, g, cfg, fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="#888")

                if ri == 0:
                    ax.set_title(str(yr), fontsize=11, fontweight="bold", pad=5)
                if ci == 0:
                    ax.set_ylabel(ss.replace(" (", "\n("), fontsize=9,
                                  fontweight="bold", labelpad=4)
                ax.set_xticks([]); ax.set_yticks([])

        cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.78])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(cfg["label"], fontsize=10, labelpad=8)
        cb.ax.tick_params(labelsize=8)

        # Annotate which sites are canal vs. bay in a small text box
        fig.text(0.01, 0.50,
                 "Canal sites:\nLR01 · MR01\nGOC-014\n\nBay sites:\nBB14 · BB25",
                 fontsize=7.5, va="center", color="#444",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7, lw=0.5))

        fig.suptitle(
            f"Biscayne Bay — {cfg['label']} · 2021–2024\n"
            "Grab-sample sites (surface)  |  Columns = year  |  Rows = season",
            fontsize=12, fontweight="bold",
        )

        out = OUT_DIR / f"{fig_start + fig_idx:02d}_map_nutrient_{grab_col}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Figure 36: Annual NH4 + DO stress — canal / bay / outfall comparison
# ---------------------------------------------------------------------------

def plot_canal_stress(grabs: pd.DataFrame):
    """
    One row per year; two columns (left=NH4, right=DO%Sat).
    Bubbles coloured by value; outline colour = site type.
    """
    all_lons = list(grabs["lon"].dropna())
    all_lats = list(grabs["lat"].dropna())
    bounds   = _bounds(all_lons, all_lats, pad_lon=0.08, pad_lat=0.05)

    years = sorted(grabs["year"].unique())

    panels = [
        ("nh4",    {"label": "NH₄ (µmol/L)",  "cmap": "YlOrRd", "vmin": 0, "vmax": 25}),
        ("do_per", {"label": "DO (%Sat)",       "cmap": "RdYlGn", "vmin": 40, "vmax": 110}),
    ]

    fig, axes = plt.subplots(
        len(years), 2,
        figsize=(2 * 4.2, len(years) * 4.0),
    )
    if len(years) == 1:
        axes = axes[np.newaxis, :]

    for ri, yr in enumerate(years):
        for ci, (grab_col, cfg) in enumerate(panels):
            ax = axes[ri, ci]
            _add_basemap(ax, bounds)

            g = grabs[grabs["year"] == yr].copy()
            for c in [grab_col]:
                g[c] = pd.to_numeric(g[c], errors="coerce")
            g = g.groupby(
                ["site_name", "lat", "lon", "site_type"]
            )[grab_col].median().reset_index().rename(columns={grab_col: "value"})

            norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
            cmap = plt.get_cmap(cfg["cmap"])

            for _, row in g.iterrows():
                x, y  = to_merc(row["lon"], row["lat"])
                color = cmap(norm(row["value"])) if not np.isnan(row["value"]) else "lightgrey"
                edge  = SITE_TYPE_COLOR.get(row["site_type"], "#333")
                ax.scatter(x, y, s=280, c=[color], marker="o",
                           edgecolors=edge, linewidths=2.0, zorder=5)

            if ri == 0:
                ax.set_title(cfg["label"], fontsize=9, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(str(yr), fontsize=9, fontweight="bold", labelpad=4)
            ax.set_xticks([]); ax.set_yticks([])

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    # Site-type legend (edge colours)
    legend_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="",
                      markersize=9, markerfacecolor="none", markeredgewidth=2,
                      label=t)
        for t, c in SITE_TYPE_COLOR.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Biscayne Bay — Annual Hypoxia Stress Indicators (2021–2024)\n"
        "Fill colour = value  |  Edge colour = site type",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = OUT_DIR / "36_map_canal_stress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading grab-sample data (Surface, 2021-2024)...")
    grabs = load_grabs()
    print(f"  {len(grabs)} rows, {grabs['site_name'].nunique()} sites, "
          f"{grabs['year'].min()}-{grabs['year'].max()}")

    print("\nGenerating combined 2021-2026 seasonal maps (temp, sal, DO)...")
    plot_combined_seasonal(grabs, fig_start=30)

    print("\nGenerating nutrient seasonal maps (pH, Chl-a, DIN)...")
    plot_nutrient_seasonal(grabs, fig_start=33)

    print("\nGenerating canal hypoxia stress map (NH4 + DO)...")
    plot_canal_stress(grabs)

    print(f"\nDone — figures 30-36 saved to {OUT_DIR}/")
