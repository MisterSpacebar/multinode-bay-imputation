"""
paper_figure.py
---------------
Two publication-quality figures for seasonal gradient analysis:

  report/paper_seasonal_maps.png   - 2-season x 3-variable compact map grid
  report/paper_gradient_profiles.png - N-S latitudinal gradient profiles

Run:
    python paper_figure.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import contextily as ctx
from pathlib import Path
from pyproj import Transformer

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

OUT_DIR     = Path("report")
GRAB_CSV    = Path("water_data/2021_to_2024/BB_thru2024.csv")
IMPUTED_DIR = Path("imputed_output")
OUT_DIR.mkdir(exist_ok=True)

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def merc(lon, lat): return _to_merc.transform(lon, lat)

# ── Sensor station metadata ────────────────────────────────────────────────

SENSOR_NODES = {
    "L0":          {"lat": 25.9114, "lon": -80.1373, "label": "L0"},
    "L1":          {"lat": 25.8749, "lon": -80.1832, "label": "L1"},
    "L2":          {"lat": 25.8537, "lon": -80.1594, "label": "L2"},
    "L6":          {"lat": 25.8442, "lon": -80.1489, "label": "L6"},
    "L7":          {"lat": 25.7791, "lon": -80.2089, "label": "L7"},
    "biscayne_bay":{"lat": 25.8704, "lon": -80.1649, "label": "Bay"},
    "consolidated_crest5": {"lat": 25.7270, "lon": -80.2697, "label": "S.Bay"},
}
SENSOR_FILES = {
    "L0":          "raw-data-platformL0_parameters_imputed.csv",
    "L1":          "raw-data-platformL1_parameters_imputed.csv",
    "L2":          "raw-data-platformL2_parameters_imputed.csv",
    "L6":          "raw-data-platformL6_parameters_imputed.csv",
    "L7":          "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay":"biscayne_bay_imputed.csv",
    "consolidated_crest5": "consolidated_crest5_imputed.csv",
}

SEASONS = {"Win (DJF)": [12,1,2], "Spr (MAM)": [3,4,5],
           "Sum (JJA)": [6,7,8],  "Aut (SON)": [9,10,11]}
KEY_SEASONS = ["Win (DJF)", "Sum (JJA)"]   # two seasons for maps

VARIABLES = {
    "temp":   {"label": "Water Temp (°C)",  "cmap": "RdYlBu_r", "vmin": 18, "vmax": 34,
               "sensor_col": "temp_c"},
    "sal":    {"label": "Salinity (ppt)",    "cmap": "YlOrBr",   "vmin": 15, "vmax": 38,
               "sensor_col": "sal_ppt"},
    "do_mgL": {"label": "DO (mg/L)",         "cmap": "RdYlGn",   "vmin": 2,  "vmax": 10,
               "sensor_col": "odo_mgL"},
}

# ── Load data ──────────────────────────────────────────────────────────────

def load_all():
    """Returns (grab_clim, sensor_clim) where each maps (season, var) -> DataFrame."""
    # --- grab samples ---
    df = pd.read_csv(GRAB_CSV, encoding="latin-1")
    df = df[df["sample_type"] == "Surface"].copy()
    df = df[~df["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["site_name"] = df["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)
    for col in ["temp","sal","do_mgL"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    site_coords = df.groupby("site_name")[["lat_dec","lon_dec"]].median()
    df["lat"] = df["site_name"].map(site_coords["lat_dec"])
    df["lon"] = df["site_name"].map(site_coords["lon_dec"])
    df["month"] = df["date"].dt.month
    m2s = {m: s for s, ms in SEASONS.items() for m in ms}
    df["season"] = df["month"].map(m2s)

    grab_clim = {}
    for var in VARIABLES:
        grp = df.groupby(["site_name","lat","lon","season"])[var].agg(
            median="median", q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75), n="count"
        ).reset_index()
        grab_clim[var] = grp

    # --- sensor stations (observed only) ---
    sensor_clim = {}
    for var, cfg in VARIABLES.items():
        sc = cfg["sensor_col"]
        records = []
        for name, meta in SENSOR_NODES.items():
            path = IMPUTED_DIR / SENSOR_FILES[name]
            if not path.exists(): continue
            d = pd.read_csv(path, index_col="datetime", parse_dates=True)
            d.index = pd.to_datetime(d.index, utc=True).tz_convert(None)
            if sc not in d.columns: continue
            obs_col = sc + "_observed"
            if obs_col in d.columns:
                d.loc[d[obs_col] == 0, sc] = np.nan
            d["month"] = d.index.month
            d["season"] = d["month"].map(m2s)
            for ss, grp in d.groupby("season")[sc]:
                vals = grp.dropna()
                if len(vals) < 10: continue
                records.append({"site_name": name, "lat": meta["lat"],
                                 "lon": meta["lon"], "season": ss,
                                 "median": vals.median(),
                                 "q25": vals.quantile(0.25),
                                 "q75": vals.quantile(0.75),
                                 "n": len(vals)})
        sensor_clim[var] = pd.DataFrame(records)

    return grab_clim, sensor_clim

# ── Figure 1: Seasonal climatology maps ───────────────────────────────────

def paper_seasonal_maps(grab_clim, sensor_clim):
    all_lons = [s["lon"] for s in SENSOR_NODES.values()]
    all_lats = [s["lat"] for s in SENSOR_NODES.values()]

    df_grab = next(iter(grab_clim.values()))
    all_lons += df_grab["lon"].dropna().tolist()
    all_lats += df_grab["lat"].dropna().tolist()

    pad = 0.045
    x0, y0 = merc(min(all_lons) - pad, min(all_lats) - pad)
    x1, y1 = merc(max(all_lons) + pad, max(all_lats) + pad)

    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    fig.subplots_adjust(wspace=0.06, hspace=0.10,
                        left=0.05, right=0.93, top=0.91, bottom=0.04)

    for ci, (var, cfg) in enumerate(VARIABLES.items()):
        norm = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
        cmap = plt.get_cmap(cfg["cmap"])
        gc   = grab_clim[var]
        sc   = sensor_clim[var]

        for ri, ss in enumerate(KEY_SEASONS):
            ax = axes[ri, ci]
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            try:
                ctx.add_basemap(ax, crs="EPSG:3857",
                                source=ctx.providers.CartoDB.Positron,
                                zoom=12, attribution=False)
            except Exception:
                ax.set_facecolor("#e8f4f8")
            ax.set_xticks([]); ax.set_yticks([])

            # Grab bubbles
            g = gc[gc["season"] == ss]
            for _, row in g.iterrows():
                if np.isnan(row["median"]): continue
                x, y = merc(row["lon"], row["lat"])
                c = cmap(norm(row["median"]))
                ax.scatter(x, y, s=180, c=[c], marker="o",
                           edgecolors="white", linewidths=1.0, zorder=5)

            # Sensor diamonds (larger, on top)
            s = sc[sc["season"] == ss]
            for _, row in s.iterrows():
                if np.isnan(row["median"]): continue
                x, y = merc(row["lon"], row["lat"])
                c = cmap(norm(row["median"]))
                ax.scatter(x, y, s=280, c=[c], marker="D",
                           edgecolors="white", linewidths=1.5, zorder=6)

            # Panel label: (a)-(f)
            label = chr(97 + ri * 3 + ci)
            ax.text(0.03, 0.97, f"({label})", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.8, lw=0))

            if ri == 0:
                ax.set_title(cfg["label"], fontsize=10, fontweight="bold", pad=5)
            if ci == 0:
                season_label = "Summer (JJA)" if "Sum" in ss else "Winter (DJF)"
                ax.set_ylabel(season_label, fontsize=10, labelpad=4)

        # Colorbar below each column
        cbar_ax = fig.add_axes([
            0.05 + ci * (0.88/3) + 0.01,
            0.01,
            0.88/3 - 0.02,
            0.018,
        ])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7.5)

    # Legend
    grab_h   = mlines.Line2D([], [], color="#555", marker="o", linestyle="",
                              markersize=7, label="Grab samples (2022-2024)")
    sensor_h = mlines.Line2D([], [], color="#555", marker="D", linestyle="",
                              markersize=8, label="Continuous sensors (2025-2026)")
    fig.legend(handles=[grab_h, sensor_h], loc="upper right",
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(
        "Seasonal climatology of surface water properties in Biscayne Bay\n"
        "Multi-year medians: grab samples 2022-2024 and continuous sensors 2025-2026",
        fontsize=10.5, fontweight="bold",
    )

    out = OUT_DIR / "paper_seasonal_maps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Figure 2: Latitudinal gradient profiles ────────────────────────────────

def paper_gradient_profiles(grab_clim, sensor_clim):
    SEASON_STYLE = {
        "Win (DJF)": dict(color="#2196F3", lw=1.8, ls="-",  label="Winter (DJF)",  marker="o"),
        "Spr (MAM)": dict(color="#4CAF50", lw=1.4, ls="--", label="Spring (MAM)", marker="s"),
        "Sum (JJA)": dict(color="#F44336", lw=1.8, ls="-",  label="Summer (JJA)", marker="^"),
        "Aut (SON)": dict(color="#FF9800", lw=1.4, ls="--", label="Autumn (SON)", marker="D"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.98, top=0.88, bottom=0.12)

    for ai, (var, cfg) in enumerate(VARIABLES.items()):
        ax   = axes[ai]
        gc   = grab_clim[var]
        sc   = sensor_clim[var]

        for ss, style in SEASON_STYLE.items():
            # Grab sites
            g = gc[gc["season"] == ss].sort_values("lat")
            if not g.empty:
                ax.plot(g["lat"], g["median"], color=style["color"],
                        lw=style["lw"], ls=style["ls"],
                        marker=style["marker"], ms=5, zorder=4,
                        label=style["label"] if ai == 0 else "_")
                ax.fill_between(g["lat"], g["q25"], g["q75"],
                                color=style["color"], alpha=0.12, zorder=3)

            # Sensor stations
            s = sc[sc["season"] == ss].sort_values("lat")
            if not s.empty:
                ax.plot(s["lat"], s["median"], color=style["color"],
                        lw=style["lw"], ls=style["ls"],
                        marker="*", ms=9, zorder=5)
                ax.fill_between(s["lat"], s["q25"], s["q75"],
                                color=style["color"], alpha=0.12, zorder=3)

        ax.set_xlabel("Latitude (°N)", fontsize=9)
        ax.set_ylabel(cfg["label"], fontsize=9)
        ax.set_title(cfg["label"], fontsize=10, fontweight="bold", pad=4)
        ax.invert_xaxis()   # north on the left
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Annotate north/south direction
        ax.text(0.02, 0.97, "N", transform=ax.transAxes,
                va="top", fontsize=8, color="#555")
        ax.text(0.96, 0.97, "S", transform=ax.transAxes,
                va="top", fontsize=8, color="#555", ha="right")

        # Panel letter
        ax.text(0.03, 0.05, f"({chr(97+ai)})", transform=ax.transAxes,
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))

    # Shared legend
    handles = [mlines.Line2D([], [], **{k:v for k,v in s.items()
                                         if k in ("color","lw","ls","label","marker")},
                              ms=5)
               for ss, s in SEASON_STYLE.items()]
    grab_h   = mlines.Line2D([], [], color="#777", marker="o", linestyle="",
                              ms=5, label="Grab sites (2022-2024)")
    sensor_h = mlines.Line2D([], [], color="#777", marker="*", linestyle="",
                              ms=8, label="Sensor stations (2025-2026)")
    fig.legend(handles=handles + [grab_h, sensor_h],
               loc="upper center", ncol=6, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.52, 1.02))

    fig.suptitle(
        "North-south latitudinal gradients in Biscayne Bay surface water properties\n"
        "Shaded bands = interquartile range  |  x-axis inverted: north (left) to south (right)",
        fontsize=9.5, y=1.10,
    )

    out = OUT_DIR / "paper_gradient_profiles.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading climatological data...")
    grab_clim, sensor_clim = load_all()

    print("Figure 1: Seasonal climatology maps...")
    paper_seasonal_maps(grab_clim, sensor_clim)

    print("Figure 2: Latitudinal gradient profiles...")
    paper_gradient_profiles(grab_clim, sensor_clim)

    print("Done.")
