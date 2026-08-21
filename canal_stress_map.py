"""
canal_stress_map.py
-------------------
Geographic stress map showing canal/outfall source sites, open-bay
receiver sites, and the two die-off event snapshots.

Three-panel figure:
  Left   - Multi-year site characterisation (mean NH4 vs mean DO%)
  Centre  - Sept 2021 event snapshot (DO% with NH4 labels)
  Right   - Oct 2022  event snapshot (DO% with DIN labels)

Output: report/canal_stress_map.png

Run:
    python canal_stress_map.py
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

OUT_DIR  = Path("report")
GRAB_CSV = Path("water_data/2021_to_2024/BB_thru2024.csv")
OUT_DIR.mkdir(exist_ok=True)

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def merc(lon, lat):
    return _to_merc.transform(lon, lat)

# ── Load and clean grab data ──────────────────────────────────────────────

df = pd.read_csv(GRAB_CSV, encoding="latin-1")
df = df[df["sample_type"] == "Surface"].copy()
# Exclude ocean-side GOC outfall/reef/inlet sites  - keep bay interior only
EXCLUDE_TYPES = {"Inlet", "Reef", "Outfall"}
df = df[~df["site_type"].isin(EXCLUDE_TYPES)].copy()
df["date"]      = pd.to_datetime(df["date"])
df["site_name"] = df["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)
for col in ["nh4", "din", "do_per", "do_mgL", "sal", "temp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Canonical coords per site
site_meta = df.groupby("site_name").agg(
    lat=("lat_dec", "median"), lon=("lon_dec", "median"),
    site_type=("site_type", "first")
).reset_index()

# ── Site-type classification ──────────────────────────────────────────────

CANAL_SITES   = {"LR01", "MR01"}
OUTFALL_SITES = {"GOC-012", "GOC-013", "GOC-014", "GOC-015"}
INLET_SITES   = {"GOC-001", "GOC-002", "GOC-003", "GOC-004"}
REEF_SITES    = {"GOC-006", "GOC-011"}
BAY_SITES     = {"BB14", "BB17", "BB18", "BB22", "BB24", "BB25", "BB26", "BB28"}

def classify(name):
    if name in CANAL_SITES:   return "Canal mouth"
    if name in OUTFALL_SITES: return "Outfall"
    if name in INLET_SITES:   return "Inlet"
    if name in REEF_SITES:    return "Reef"
    if name in BAY_SITES:     return "Open bay"
    return "Other"

site_meta["category"] = site_meta["site_name"].apply(classify)

CATEGORY_STYLE = {
    "Canal mouth": dict(marker="v", color="#d32f2f", zorder=7, s_base=400),
    "Outfall":     dict(marker="^", color="#f57c00", zorder=7, s_base=300),
    "Inlet":       dict(marker="s", color="#7b1fa2", zorder=6, s_base=200),
    "Reef":        dict(marker="D", color="#1565c0", zorder=6, s_base=200),
    "Open bay":    dict(marker="o", color="#2e7d32", zorder=5, s_base=220),
    "Other":       dict(marker="o", color="#78909c", zorder=4, s_base=150),
}

# ── Multi-year means (all years, 2021-2024) ───────────────────────────────

annual = df.groupby("site_name").agg(
    mean_nh4=("nh4",    "median"),
    mean_do =("do_per", "median"),
).reset_index()
annual = annual.merge(site_meta, on="site_name")

# ── Event snapshots ───────────────────────────────────────────────────────

def event_data(year, month):
    ev = df[(df["date"].dt.year == year) & (df["date"].dt.month == month)].copy()
    ev = ev.drop_duplicates("site_name")
    return ev.merge(site_meta, on="site_name", how="left")

ev_2021 = event_data(2021,  9)
ev_2022 = event_data(2022, 10)

# ── Basemap bounds ────────────────────────────────────────────────────────

all_lons = site_meta["lon"].dropna().tolist()
all_lats = site_meta["lat"].dropna().tolist()
pad_lon, pad_lat = 0.055, 0.040
x0, y0 = merc(min(all_lons) - pad_lon, min(all_lats) - pad_lat)
x1, y1 = merc(max(all_lons) + pad_lon, max(all_lats) + pad_lat)
BOUNDS = (x0, y0, x1, y1)

def add_basemap(ax):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=12, attribution=False)
    except Exception:
        ax.set_facecolor("#d4e9f7")
    ax.set_xticks([]); ax.set_yticks([])

# ── Drawing helpers ───────────────────────────────────────────────────────

def plot_site_bubbles(ax, data, value_col, cmap, norm, label_sites=None,
                      label_col=None, size_col=None, base_size=220):
    """Scatter all sites; colour by value_col; label key sites."""
    cm  = plt.get_cmap(cmap)
    for _, row in data.iterrows():
        name  = row["site_name"]
        style = CATEGORY_STYLE.get(row.get("category", "Other"),
                                    CATEGORY_STYLE["Other"])
        x, y  = merc(row["lon"], row["lat"])
        v     = row.get(value_col, np.nan)
        c     = cm(norm(v)) if not np.isnan(v) else "lightgrey"
        sz    = base_size if size_col is None else float(row.get(size_col, base_size))
        ax.scatter(x, y, s=sz, c=[c], marker=style["marker"],
                   edgecolors="white", linewidths=1.5, zorder=style["zorder"])
        if label_sites and name in label_sites:
            lval = row.get(label_col or value_col, np.nan)
            txt  = f"{name}\n{lval:.1f}" if not np.isnan(lval) else name
            # White halo then coloured text
            ax.annotate(txt, (x, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=6.5, fontweight="bold", color="#222", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.75, lw=0))

# ── Figure ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 10))
fig.subplots_adjust(wspace=0.06, left=0.02, right=0.92, top=0.88, bottom=0.05)

LABEL_SITES = CANAL_SITES | OUTFALL_SITES | {"BB14", "BB25"}

# ──────────────────────────────────────────────────────
# Panel 1  - multi-year site characterisation
# Bubble colour = median DO%; size proportional to median NH4
# ──────────────────────────────────────────────────────
ax = axes[0]
add_basemap(ax)

do_norm  = mcolors.Normalize(vmin=50, vmax=105)
nh4_data = annual.copy()
nh4_max  = nh4_data["mean_nh4"].replace(0, np.nan).quantile(0.95)
# Scale bubble size: 150 to 700 proportional to NH4
nh4_data["sz"] = 150 + (nh4_data["mean_nh4"].clip(0, nh4_max) / nh4_max) * 550

plot_site_bubbles(ax, nh4_data, "mean_do", "RdYlGn", do_norm,
                  label_sites=LABEL_SITES, label_col="mean_nh4",
                  size_col="sz")

ax.set_title("Multi-year median\n(2021–2024)", fontsize=11, fontweight="bold", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)\nBubble size ∝ NH₄",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

# DO% colourbar
sm_do = plt.cm.ScalarMappable(cmap="RdYlGn", norm=do_norm)
sm_do.set_array([])
cb1 = fig.colorbar(sm_do, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
cb1.set_label("DO (% Sat)", fontsize=8)
cb1.ax.tick_params(labelsize=7)

# ──────────────────────────────────────────────────────
# Panel 2  - Sept 2021 event
# Colour = DO%; labels show NH4 at key sites
# Red star overlay on sites with DO < 70%
# ──────────────────────────────────────────────────────
ax = axes[1]
add_basemap(ax)

plot_site_bubbles(ax, ev_2021, "do_per", "RdYlGn", do_norm,
                  label_sites=LABEL_SITES, label_col="nh4",
                  base_size=320)

# Stress markers at sites with DO < 70 %
for _, row in ev_2021.iterrows():
    if row.get("do_per", 100) < 70:
        x, y = merc(row["lon"], row["lat"])
        ax.scatter(x, y, s=600, c="none", marker="*",
                   edgecolors="#b71c1c", linewidths=1.5, zorder=8)

ax.set_title("★ Sept 2021 event\n(Fish + Seagrass die-off)", fontsize=11,
             fontweight="bold", color="#b71c1c", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)\nLabels = NH₄ (µmol/L)\n★ = DO < 70%",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

fig.colorbar(sm_do, ax=ax, fraction=0.04, pad=0.02, shrink=0.7).ax.tick_params(labelsize=7)

# ──────────────────────────────────────────────────────
# Panel 3  - Oct 2022 event
# Colour = DO%; labels show DIN at key sites
# ──────────────────────────────────────────────────────
ax = axes[2]
add_basemap(ax)

plot_site_bubbles(ax, ev_2022, "do_per", "RdYlGn", do_norm,
                  label_sites=LABEL_SITES, label_col="din",
                  base_size=320)

for _, row in ev_2022.iterrows():
    if row.get("do_per", 100) < 70:
        x, y = merc(row["lon"], row["lat"])
        ax.scatter(x, y, s=600, c="none", marker="*",
                   edgecolors="#b71c1c", linewidths=1.5, zorder=8)

ax.set_title("★ Oct 2022 event\n(Seagrass die-off)", fontsize=11,
             fontweight="bold", color="#b71c1c", pad=6)
ax.text(0.03, 0.97,
        "Colour = DO (%Sat)\nLabels = DIN (µmol/L)\n★ = DO < 70%",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0.5))

fig.colorbar(sm_do, ax=ax, fraction=0.04, pad=0.02, shrink=0.7).ax.tick_params(labelsize=7)

# ── Shared marker legend ──────────────────────────────────────────────────

legend_handles = []
for cat, style in CATEGORY_STYLE.items():
    if cat == "Other":
        continue
    legend_handles.append(
        mlines.Line2D([], [], color=style["color"], marker=style["marker"],
                      linestyle="", markersize=9, markeredgecolor="white",
                      markeredgewidth=0.8, label=cat)
    )
legend_handles.append(
    mlines.Line2D([], [], color="#b71c1c", marker="*", linestyle="",
                  markersize=10, markerfacecolor="none",
                  markeredgewidth=1.5, label="DO < 70% (hypoxia)")
)
legend_handles.append(
    mlines.Line2D([], [], color="lightgrey", marker="o", linestyle="",
                  markersize=8, label="No sample this month")
)

fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
           fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.47, -0.01))

fig.suptitle(
    "Biscayne Bay  - Canal/Outfall Stress Map\n"
    "Source-adjacent (canal/outfall) vs open-bay sites  |  "
    "Event stress locations highlighted",
    fontsize=13, fontweight="bold",
)

out = OUT_DIR / "canal_stress_map.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
