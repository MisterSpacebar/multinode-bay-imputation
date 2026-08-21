"""
canal_stress_map_v2.py
----------------------
Canal/outfall stress map with sensor stations added to show the
northern bay (north of JFK Causeway) that has no grab-sample coverage.

Grab sites (2021-2024):  circles  - coloured by value
Sensor stations (2025+): squares  - coloured by value where available,
                                    hollow where not deployed at event time

Panels:
  Left   - Multi-year site characterisation (grab median + sensor 2025-2026)
  Centre - Sept 2021 event  (sensor stations shown hollow: not yet deployed)
  Right  - Oct 2022 event   (same)

Output: report/canal_stress_map_v2.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import contextily as ctx
from pathlib import Path
from pyproj import Transformer

OUT_DIR     = Path("report")
GRAB_CSV    = Path("water_data/2021_to_2024/BB_thru2024.csv")
IMPUTED_DIR = Path("imputed_output")
OUT_DIR.mkdir(exist_ok=True)

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def merc(lon, lat):
    return _to_merc.transform(lon, lat)

# ── Sensor station metadata ────────────────────────────────────────────────

SENSOR_META = {
    "L0":          {"lat": 25.9114, "lon": -80.1373, "label": "L0\n(Haulover)"},
    "L1":          {"lat": 25.8749, "lon": -80.1832, "label": "L1\n(B.Canal)"},
    "biscayne_bay":{"lat": 25.8704, "lon": -80.1649, "label": "Bay\n(p2)"},
    "L2":          {"lat": 25.8537, "lon": -80.1594, "label": "L2\n(Little R.)"},
}
IMPUTED_FILES = {
    "L0":          "raw-data-platformL0_parameters_imputed.csv",
    "L1":          "raw-data-platformL1_parameters_imputed.csv",
    "biscayne_bay":"biscayne_bay_imputed.csv",
    "L2":          "raw-data-platformL2_parameters_imputed.csv",
}

# ── Load and clean grab data ───────────────────────────────────────────────

df = pd.read_csv(GRAB_CSV, encoding="latin-1")
df = df[df["sample_type"] == "Surface"].copy()
df = df[~df["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()
df["date"]      = pd.to_datetime(df["date"])
df["site_name"] = df["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)
for col in ["nh4", "din", "do_per", "do_mgL", "sal", "temp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

site_meta = df.groupby("site_name").agg(
    lat=("lat_dec", "median"), lon=("lon_dec", "median"),
    site_type=("site_type", "first")
).reset_index()

def classify(name):
    CANAL   = {"LR01", "MR01"}
    BAY     = {"BB14","BB17","BB18","BB22","BB24","BB25","BB26","BB28"}
    if name in CANAL:   return "Canal mouth"
    if name in BAY:     return "Open bay"
    return "Other"

site_meta["category"] = site_meta["site_name"].apply(classify)

GRAB_STYLE = {
    "Canal mouth": dict(marker="v", color="#d32f2f", s_base=420, zorder=7),
    "Open bay":    dict(marker="o", color="#2e7d32", s_base=220, zorder=5),
    "Other":       dict(marker="o", color="#78909c", s_base=150, zorder=4),
}

# ── Load sensor annual DO% mean (2025-2026) ────────────────────────────────

def sensor_annual_do():
    """Annual mean observed DO%Sat for each sensor station."""
    results = {}
    for name, fname in IMPUTED_FILES.items():
        path = IMPUTED_DIR / fname
        if not path.exists():
            continue
        d = pd.read_csv(path, index_col="datetime", parse_dates=True)
        d.index = pd.to_datetime(d.index, utc=True).tz_convert(None)
        if "odo_pct" not in d.columns:
            continue
        obs_col = "odo_pct_observed"
        if obs_col in d.columns:
            d.loc[d[obs_col] == 0, "odo_pct"] = np.nan
        results[name] = d["odo_pct"].median()
    return results

sensor_do = sensor_annual_do()

# ── Multi-year grab medians ────────────────────────────────────────────────

annual = df.groupby("site_name").agg(
    mean_nh4=("nh4",    "median"),
    mean_do =("do_per", "median"),
).reset_index().merge(site_meta, on="site_name")
nh4_max = annual["mean_nh4"].replace(0, np.nan).quantile(0.95)
annual["sz"] = 150 + (annual["mean_nh4"].clip(0, nh4_max) / nh4_max) * 550

# ── Event grab snapshots ───────────────────────────────────────────────────

def event_data(year, month):
    ev = df[(df["date"].dt.year == year) & (df["date"].dt.month == month)].copy()
    return ev.drop_duplicates("site_name").merge(site_meta, on="site_name", how="left")

ev_2021 = event_data(2021, 9)
ev_2022 = event_data(2022, 10)

# ── Basemap bounds (expanded north to include sensor stations) ─────────────

all_lons = site_meta["lon"].tolist() + [m["lon"] for m in SENSOR_META.values()]
all_lats = site_meta["lat"].tolist() + [m["lat"] for m in SENSOR_META.values()]
x0, y0 = merc(min(all_lons) - 0.055, min(all_lats) - 0.035)
x1, y1 = merc(max(all_lons) + 0.055, max(all_lats) + 0.035)

def add_basemap(ax):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=12, attribution=False)
    except Exception:
        ax.set_facecolor("#d4e9f7")
    ax.set_xticks([]); ax.set_yticks([])

JFK_LAT = 25.853
JFK_Y   = merc(-80.20, JFK_LAT)[1]  # mercator y for the causeway line

# ── Drawing helpers ────────────────────────────────────────────────────────

DO_NORM  = mcolors.Normalize(vmin=50, vmax=105)
DO_CMAP  = "RdYlGn"
LABEL_SITES = {"LR01", "MR01", "BB14", "BB25"}


def draw_grab_bubbles(ax, data, value_col, label_col=None, size_col=None):
    cm = plt.get_cmap(DO_CMAP)
    for _, row in data.iterrows():
        style = GRAB_STYLE.get(row.get("category","Other"), GRAB_STYLE["Other"])
        x, y  = merc(row["lon"], row["lat"])
        v     = row.get(value_col, np.nan)
        c     = cm(DO_NORM(v)) if not np.isnan(v) else "lightgrey"
        sz    = float(row.get(size_col, style["s_base"])) if size_col else style["s_base"]
        ax.scatter(x, y, s=sz, c=[c], marker=style["marker"],
                   edgecolors="white", linewidths=1.5, zorder=style["zorder"])
        name = row["site_name"]
        if name in LABEL_SITES:
            lv = row.get(label_col or value_col, np.nan)
            ax.annotate(f"{name}\n{lv:.1f}" if not np.isnan(lv) else name,
                        (x, y), xytext=(5, 6), textcoords="offset points",
                        fontsize=6.5, fontweight="bold", color="#222", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.75, lw=0))


def draw_sensor_stations(ax, do_values=None, hollow=False):
    """Plot sensor stations as squares; hollow=True when not yet deployed."""
    cm = plt.get_cmap(DO_CMAP)
    for name, meta in SENSOR_META.items():
        x, y = merc(meta["lon"], meta["lat"])
        v    = (do_values or {}).get(name, np.nan)
        if hollow or np.isnan(v):
            ax.scatter(x, y, s=320, c="none", marker="s",
                       edgecolors="#1565c0", linewidths=2.0, zorder=8,
                       linestyle="--")
        else:
            c = cm(DO_NORM(v))
            ax.scatter(x, y, s=320, c=[c], marker="s",
                       edgecolors="#1565c0", linewidths=2.0, zorder=8)
        ax.annotate(meta["label"], (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=6, color="#1565c0",
                    fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              alpha=0.7, lw=0))


def draw_jfk_line(ax):
    ax.axhline(JFK_Y, color="#555", lw=1.2, ls="--", alpha=0.7, zorder=10)
    ax.text(x0 + (x1-x0)*0.02, JFK_Y + (y1-y0)*0.01,
            "JFK Causeway", fontsize=6.5, color="#555",
            fontstyle="italic", zorder=11)


def draw_gap_annotation(ax):
    gap_y = merc(-80.18, (JFK_LAT + 25.845)/2)[1]
    ax.annotate("monitoring\ngap", xy=(merc(-80.18, JFK_LAT)[0], JFK_Y),
                xytext=(merc(-80.185, (JFK_LAT + 25.845)/2)[0], gap_y),
                fontsize=6, color="#b71c1c", ha="center",
                arrowprops=dict(arrowstyle="-", color="#b71c1c",
                                lw=0.8, linestyle="dotted"),
                zorder=11)


# ── Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 12))
fig.subplots_adjust(wspace=0.06, left=0.02, right=0.92, top=0.88, bottom=0.07)

# ── Panel 1: multi-year median + sensor 2025-2026 ─────────────────────────
ax = axes[0]
add_basemap(ax)
draw_jfk_line(ax)
draw_gap_annotation(ax)
draw_grab_bubbles(ax, annual, "mean_do", label_col="mean_nh4", size_col="sz")
draw_sensor_stations(ax, do_values=sensor_do)
ax.set_title("Multi-year median\nGrab: 2021-2024  |  Sensors: 2025-2026",
             fontsize=10, fontweight="bold", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)\nGrab bubble size prop. NH4\n■ = sensor station (DO 2025-2026)",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

# ── Panel 2: Sept 2021 event ───────────────────────────────────────────────
ax = axes[1]
add_basemap(ax)
draw_jfk_line(ax)
draw_gap_annotation(ax)
draw_grab_bubbles(ax, ev_2021, "do_per", label_col="nh4")
draw_sensor_stations(ax, hollow=True)   # not yet deployed in 2021
for _, row in ev_2021.iterrows():
    if row.get("do_per", 100) < 70:
        x, y = merc(row["lon"], row["lat"])
        ax.scatter(x, y, s=600, c="none", marker="*",
                   edgecolors="#b71c1c", linewidths=1.5, zorder=9)
ax.set_title("Sept 2021 event\n(Fish + Seagrass die-off)",
             fontsize=10, fontweight="bold", color="#b71c1c", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)  |  Labels = NH4\n★ = DO < 70%\n□ = sensor (not deployed 2021)",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

# ── Panel 3: Oct 2022 event ────────────────────────────────────────────────
ax = axes[2]
add_basemap(ax)
draw_jfk_line(ax)
draw_gap_annotation(ax)
draw_grab_bubbles(ax, ev_2022, "do_per", label_col="din")
draw_sensor_stations(ax, hollow=True)   # not yet deployed in 2022
for _, row in ev_2022.iterrows():
    if row.get("do_per", 100) < 70:
        x, y = merc(row["lon"], row["lat"])
        ax.scatter(x, y, s=600, c="none", marker="*",
                   edgecolors="#b71c1c", linewidths=1.5, zorder=9)
ax.set_title("Oct 2022 event\n(Seagrass die-off)",
             fontsize=10, fontweight="bold", color="#b71c1c", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)  |  Labels = DIN\n★ = DO < 70%\n□ = sensor (not deployed 2022)",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

# ── Shared colorbar ─────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=DO_CMAP, norm=DO_NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01, shrink=0.75)
cbar.set_label("DO (% Sat)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── Legend ───────────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color="#d32f2f", marker="v", linestyle="",
                  markersize=9, markeredgecolor="white", label="Canal mouth (LR01, MR01)"),
    mlines.Line2D([], [], color="#2e7d32", marker="o", linestyle="",
                  markersize=9, markeredgecolor="white", label="Open bay (BB sites)"),
    mlines.Line2D([], [], color="#1565c0", marker="s", linestyle="",
                  markersize=9, markerfacecolor="steelblue",
                  markeredgewidth=2, label="Sensor station (2025-2026, filled = DO data)"),
    mlines.Line2D([], [], color="#1565c0", marker="s", linestyle="",
                  markersize=9, markerfacecolor="none",
                  markeredgewidth=2, label="Sensor station (hollow = not deployed at event)"),
    mlines.Line2D([], [], color="#b71c1c", marker="*", linestyle="",
                  markersize=10, markerfacecolor="none",
                  markeredgewidth=1.5, label="DO < 70% (hypoxic threshold)"),
    mlines.Line2D([], [], color="#555", marker="|", linestyle="--",
                  markersize=10, label="JFK (79th St) Causeway"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.46, -0.02))

fig.suptitle(
    "Biscayne Bay - Canal/Outfall Stress Map v2\n"
    "Northern bay (above JFK Causeway) covered by sensors 2025-2026 only; "
    "no grab-sample data available for 2021-2024 in that zone",
    fontsize=12, fontweight="bold",
)

out = OUT_DIR / "canal_stress_map_v2.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")
