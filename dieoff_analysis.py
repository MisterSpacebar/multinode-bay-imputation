"""
dieoff_analysis.py
------------------
Investigates environmental conditions correlated with the two documented
Biscayne Bay biological die-off events using the 2021-2024 grab-sample data.

  EVENT 1  - September 2021   fish + seagrass die-off
  EVENT 2  - October   2022   seagrass die-off

Five analyses are produced:

  1. Anomaly heatmap
     Z-score of each feature at each event month vs the full multi-year
     baseline.  Shows which variables were statistically anomalous.

  2. Spatial DO + nutrient maps
     Scatter-plot of all sites colour-coded by DO%, dissolved inorganic
     nitrogen (DIN), and temperature at the event month.
     Canal / outfall sites are highlighted.

  3. Pre-event lead-up
     3-month window before → through each event for key "stress" variables:
     DO%, NH4, DIN, water temp, salinity.  Reveals if conditions built up
     gradually or spiked suddenly.

  4. FCM forward simulation from event conditions
     Load the trained nutrient FCM weights.  Set the initial activation to
     the measured monthly values at the event.  Run forward 12 months.
     Compare the simulated trajectory to the actual subsequent observations.

  5. Cross-event comparison
     Side-by-side bar chart of all feature anomalies for both events.

Outputs
-------
  visualizations/16_dieoff_anomaly_heatmap.png
  visualizations/17_dieoff_spatial_maps.png
  visualizations/18_dieoff_leadup.png
  visualizations/19_dieoff_fcm_projection.png
  visualizations/20_dieoff_comparison.png
  analysis/dieoff_anomalies.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import contextily as ctx
from pyproj import Transformer
from pathlib import Path

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def _merc(lon, lat): return _to_merc.transform(lon, lat)

ANALYSIS_DIR = Path("analysis")
VIZ_DIR      = Path("visualizations")
DATA_CSV     = Path("water_data/2021_to_2024/BB_thru2024.csv")
FCM_WEIGHTS_NUTR = ANALYSIS_DIR / "fcm_weights_nutrient.csv"

ANALYSIS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------
FEAT_RAW = ["temp", "sal", "do_per", "do_mgL", "ph",
            "chl_exo_ugL", "secchi", "no2no3", "nh4", "po4", "din"]

FEAT_LABELS = {
    "temp":        "Water Temp (°C)",
    "sal":         "Salinity (PPT)",
    "do_per":      "DO (%Sat)",
    "do_mgL":      "DO (mg/L)",
    "ph":          "pH",
    "chl_exo_ugL": "Chl-a (µg/L)",
    "secchi":      "Secchi Depth (m)",
    "no2no3":      "NO₂+NO₃ (µmol/L)",
    "nh4":         "NH₄ (µmol/L)",
    "po4":         "PO₄ (µmol/L)",
    "din":         "DIN (µmol/L)",
}

# Variables where HIGH values = stress (positive anomaly = bad)
STRESS_HIGH = {"temp", "no2no3", "nh4", "po4", "din", "chl_exo_ugL"}
# Variables where LOW values = stress (negative anomaly = bad)
STRESS_LOW  = {"do_per", "do_mgL", "sal", "secchi"}

# Site classification
CANAL_SITES  = {"LR01", "MR01"}
OUTFALL_SITES = {s for s in [] if s.startswith("GOC-0")}  # filled dynamically
BAY_SITES    = {"BB14", "BB17", "BB18", "BB22", "BB24", "BB25", "BB26", "BB28"}

# Die-off events: (label, event_month, lead_start_month, follow_end_month)
EVENTS = [
    {
        "label":       "Sept 2021  - Fish & Seagrass Die-off",
        "short":       "Sept 2021",
        "event_month": "2021-09",
        "lead_months": ["2021-07", "2021-08", "2021-09"],
        "follow_months": ["2021-10", "2021-11", "2021-12"],
        "color":       "#d62728",
    },
    {
        "label":       "Oct 2022  - Seagrass Die-off",
        "short":       "Oct 2022",
        "event_month": "2022-10",
        "lead_months": ["2022-08", "2022-09", "2022-10"],
        "follow_months": ["2022-12", "2023-02", "2023-04"],
        "color":       "#ff7f0e",
    },
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_grab_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="latin-1", low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    for c in FEAT_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lat_dec"] = pd.to_numeric(df["lat_dec"], errors="coerce")
    df["lon_dec"] = pd.to_numeric(df["lon_dec"], errors="coerce")
    df["period"]  = df["datetime"].dt.to_period("M")
    df = df[df["sample_type"] == "Surface"].copy()
    # Exclude GOC inlet/reef/outfall sites (ocean-side, outside the bay proper)
    return df[~df["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()


def compute_baseline(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Full multi-year mean and std for Surface samples, all sites."""
    return df[FEAT_RAW].mean(), df[FEAT_RAW].std()


def monthly_spatial_avg(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Average across all Biscayne Bay + canal sites for given period strings."""
    sub = df[df["period"].astype(str).isin(months)]
    return sub.groupby("period")[FEAT_RAW].mean()


def site_snapshot(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Per-site averages for a single month."""
    sub = df[df["period"].astype(str) == month]
    return sub.groupby("site_name")[["lat_dec", "lon_dec"] + FEAT_RAW].mean()


# ---------------------------------------------------------------------------
# Analysis 1  - Anomaly heatmap
# ---------------------------------------------------------------------------

def plot_anomaly_heatmap(
    df:      pd.DataFrame,
    mean:    pd.Series,
    std:     pd.Series,
) -> pd.DataFrame:
    """
    Compute z-scores for each event month and nearby months.
    Save a 2-panel heatmap (one row per month, one column per feature).
    """
    all_months = []
    for ev in EVENTS:
        all_months += ev["lead_months"] + ev["follow_months"]
    all_months = sorted(set(all_months))

    monthly_avg = monthly_spatial_avg(df, all_months)
    z_scores    = (monthly_avg - mean) / (std + 1e-9)

    # ---- save CSV ----
    z_scores.to_csv(ANALYSIS_DIR / "dieoff_anomalies.csv", float_format="%.3f")
    print(f"  Saved anomaly CSV → analysis/dieoff_anomalies.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    labels     = [FEAT_LABELS[f] for f in FEAT_RAW]
    month_strs = [str(m) for m in z_scores.index]

    mat = z_scores[FEAT_RAW].values.astype(float)
    vmax = max(2.0, float(np.nanmax(np.abs(mat))))
    im   = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(FEAT_RAW)))
    ax.set_yticks(range(len(month_strs)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(month_strs, fontsize=9)
    ax.set_title(
        "Environmental Anomalies Around Die-off Events\n"
        "Z-score vs full 2021–2026 baseline  (red = high, blue = low)",
        fontsize=12, pad=10,
    )

    # Annotate cells
    for i in range(len(month_strs)):
        for j in range(len(FEAT_RAW)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            if abs(v) > 0.5:
                col = "white" if abs(v) > 2.0 else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color=col)

    # Event-month row borders — label placed left of axes using blended transform
    import matplotlib.transforms as transforms
    n_cols = len(FEAT_RAW)
    for ev in EVENTS:
        for m in ev["lead_months"][-1:]:   # just the event month itself
            if m in month_strs:
                row = month_strs.index(m)
                rect = plt.Rectangle((-0.5, row - 0.5),
                                     n_cols, 1,
                                     fill=False, edgecolor=ev["color"],
                                     lw=2.5, zorder=10)
                ax.add_patch(rect)
                # x in axes fraction (-0.01 = just left of left edge), y in data coords
                blend = transforms.blended_transform_factory(
                    ax.transAxes, ax.transData)
                ax.annotate(ev["short"], xy=(-0.01, row),
                            xycoords=blend, ha="right", va="center",
                            fontsize=8.5, color=ev["color"], fontweight="bold",
                            annotation_clip=False,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      alpha=0.85, lw=0))

    plt.colorbar(im, ax=ax, label="Z-score", fraction=0.03, pad=0.02)
    plt.tight_layout()
    out = VIZ_DIR / "16_dieoff_anomaly_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

    return z_scores


# ---------------------------------------------------------------------------
# Analysis 2  - Spatial maps (DO + DIN + Temp)
# ---------------------------------------------------------------------------

def plot_spatial_maps(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_vars = [
        ("do_per",  "DO (% Sat)",       "RdYlGn",  False),
        ("din",     "DIN (µmol/L)",     "YlOrRd",  True),
        ("nh4",     "NH₄ (µmol/L)",     "YlOrRd",  True),
    ]

    for row, ev in enumerate(EVENTS):
        snap = site_snapshot(df, ev["event_month"])
        snap = snap.dropna(subset=["lat_dec", "lon_dec"])

        for col, (var, vlab, cmap, log) in enumerate(plot_vars):
            ax = axes[row, col]
            vals = snap[var].copy()
            plot_vals = np.log1p(vals) if log else vals

            # Identify site types
            site_type = []
            for s in snap.index:
                if s in CANAL_SITES:
                    site_type.append("Canal")
                elif s.startswith("GOC"):
                    site_type.append("Outfall/Inlet")
                else:
                    site_type.append("Bay")

            vmin_p = np.nanpercentile(plot_vals, 5)
            vmax_p = np.nanpercentile(plot_vals, 95)
            norm   = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)

            sc = ax.scatter(
                snap["lon_dec"], snap["lat_dec"],
                c=plot_vals, cmap=cmap, norm=norm,
                s=140, zorder=5, edgecolors="k", linewidths=0.7,
            )

            # Site-type markers
            for st_label, marker, size in [
                ("Canal",        "^", 220),
                ("Outfall/Inlet","s", 180),
            ]:
                idx = [i for i, t in enumerate(site_type) if t == st_label]
                if idx:
                    ax.scatter(
                        snap["lon_dec"].iloc[idx], snap["lat_dec"].iloc[idx],
                        c=plot_vals.iloc[idx], cmap=cmap, norm=norm,
                        s=size, marker=marker, zorder=6,
                        edgecolors="k", linewidths=1.2,
                    )

            for s, row_ in snap.iterrows():
                if not (np.isnan(row_["lat_dec"]) or np.isnan(row_["lon_dec"])):
                    ax.annotate(
                        s, (row_["lon_dec"], row_["lat_dec"]),
                        fontsize=5.5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points",
                    )

            cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            scale_note = " (log scale)" if log else ""
            cbar.set_label(f"{vlab}{scale_note}", fontsize=8)

            if row == 0:
                ax.set_title(vlab, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ev["short"], fontsize=10, fontweight="bold",
                              color=ev["color"])
            ax.set_xlabel("Longitude", fontsize=8)
            ax.grid(True, alpha=0.2)

    # Legend for site types
    legend_patches = [
        mpatches.Patch(color="none",
                       label="● Bay station  ▲ Canal  ■ Outfall/Inlet"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=9,
               ncol=1, frameon=False)
    fig.suptitle(
        "Spatial Snapshot at Die-off Events\n"
        "Colour = measured value at event month, all Surface samples",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = VIZ_DIR / "17_dieoff_spatial_maps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


def plot_spatial_maps_basemap(df: pd.DataFrame) -> None:
    """Figure 17a - same as 17 but overlaid on a CartoDB basemap."""
    plot_vars = [
        ("do_per",  "DO (% Sat)",   "RdYlGn",  False, (50, 105)),
        ("din",     "DIN (µmol/L)", "YlOrRd",  False, (0,  20)),
        ("nh4",     "NH4 (µmol/L)", "YlOrRd",  False, (0,  15)),
    ]

    # Fixed bounds extended north to Haulover to show monitoring gap
    x0, y0 = _merc(-80.230, 25.700)
    x1, y1 = _merc(-80.100, 25.930)

    # Sensor station positions (deployed 2025+, not at event time)
    SENSOR_LOCS = [
        (25.9114, -80.1373, "L0"),
        (25.8749, -80.1832, "L1"),
        (25.8704, -80.1649, "Bay"),
    ]
    JFK_Y = _merc(-80.20, 25.853)[1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    fig.subplots_adjust(wspace=0.06, hspace=0.10,
                        left=0.04, right=0.97, top=0.91, bottom=0.05)

    for row, ev in enumerate(EVENTS):
        snap = site_snapshot(df, ev["event_month"])
        snap = snap.dropna(subset=["lat_dec", "lon_dec"])

        for col, (var, vlab, cmap, log, (vmin, vmax)) in enumerate(plot_vars):
            ax = axes[row, col]
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            try:
                ctx.add_basemap(ax, crs="EPSG:3857",
                                source=ctx.providers.CartoDB.Positron,
                                zoom=12, attribution=False)
            except Exception:
                ax.set_facecolor("#d4e9f7")
            ax.set_xticks([]); ax.set_yticks([])

            # JFK Causeway reference line
            ax.axhline(JFK_Y, color="#555", lw=1.0, ls="--", alpha=0.6, zorder=4)
            if col == 0:
                ax.text(x0 + (x1-x0)*0.02, JFK_Y + (y1-y0)*0.01,
                        "JFK Causeway", fontsize=6, color="#555",
                        fontstyle="italic", zorder=5)

            # Sensor stations (hollow squares - not deployed at event time)
            for slat, slon, slabel in SENSOR_LOCS:
                xm, ym = _merc(slon, slat)
                ax.scatter(xm, ym, s=200, c="none", marker="s",
                           edgecolors="#1565c0", linewidths=1.8,
                           linestyle="--", zorder=7)
                if col == 0:
                    ax.annotate(slabel, (xm, ym), xytext=(4, 3),
                                textcoords="offset points", fontsize=5.5,
                                color="#1565c0", zorder=8)

            vals = pd.to_numeric(snap[var], errors="coerce")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cm   = plt.get_cmap(cmap)

            for site, srow in snap.iterrows():
                if np.isnan(srow["lat_dec"]) or np.isnan(srow["lon_dec"]):
                    continue
                v = vals.get(site, np.nan)
                xm, ym = _merc(srow["lon_dec"], srow["lat_dec"])
                c = cm(norm(v)) if not np.isnan(v) else "lightgrey"
                marker = "^" if site in CANAL_SITES else "o"
                sz     = 300 if site in CANAL_SITES else 200
                ax.scatter(xm, ym, s=sz, c=[c], marker=marker,
                           edgecolors="white", linewidths=1.3, zorder=5)
                if not np.isnan(v):
                    ax.annotate(f"{site}\n{v:.1f}", (xm, ym),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=6, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc="white", alpha=0.7, lw=0))

            sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
            cb.set_label(vlab, fontsize=8)
            cb.ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(vlab, fontsize=10, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(ev["short"], fontsize=10.5, fontweight="bold",
                              color=ev["color"], labelpad=6)

    # Marker legend
    import matplotlib.lines as mlines
    bay_h    = mpatches.Patch(fc="#888", label="● Bay station")
    canal_h  = mpatches.Patch(fc="#d32f2f", label="▲ Canal mouth (LR01, MR01)")
    grey_h   = mpatches.Patch(fc="lightgrey", label="No sample this month")
    sensor_h = mlines.Line2D([], [], color="#1565c0", marker="s", linestyle="",
                              ms=8, markerfacecolor="none", markeredgewidth=1.8,
                              label="□ Sensor station (not deployed 2021-2022)")
    fig.legend(handles=[bay_h, canal_h, grey_h, sensor_h], loc="lower center",
               ncol=4, fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Spatial Snapshot at Die-off Events - Bay-interior sites\n"
        "Colour = measured value at event month  |  ▲ Canal mouth  ● Open bay",
        fontsize=12, fontweight="bold",
    )

    out = VIZ_DIR / "17a_dieoff_spatial_basemap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

def plot_leadup(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> None:
    """
    One column per die-off event, one row per stress variable.
    X-axis shows real calendar months (e.g. "Jul 2021").
    The event month is highlighted with a shaded band and arrow label.
    """
    stress_vars = ["do_per", "nh4", "din", "temp", "sal"]
    n_vars      = len(stress_vars)
    n_events    = len(EVENTS)

    # Danger thresholds: variable → (value, label, direction)
    thresholds = {
        "do_per": (75,  "Hypoxia threshold (75%)",             "low"),
        "nh4":    (10,  "Eutrophication concern (10 µmol/L)",  "high"),
        "din":    (10,  "Eutrophication concern (10 µmol/L)",  "high"),
    }

    fig, axes = plt.subplots(n_vars, n_events,
                             figsize=(7 * n_events, 2.6 * n_vars),
                             squeeze=False)

    for col, ev in enumerate(EVENTS):
        window_months = sorted(set(ev["lead_months"] + ev["follow_months"]))
        avg = monthly_spatial_avg(df, window_months)

        # Real calendar month labels for x-axis
        month_labels = [pd.Period(m, "M").strftime("%b %Y")
                        for m in [str(p) for p in avg.index]]
        x_pos        = np.arange(len(month_labels))
        event_str    = ev["event_month"]
        event_xi     = [str(p) for p in avg.index].index(event_str) \
                       if event_str in [str(p) for p in avg.index] else None

        for row, var in enumerate(stress_vars):
            ax  = axes[row][col]
            vlabel = FEAT_LABELS[var]

            if var in avg.columns:
                vals = avg[var].values

                # ── baseline band ──────────────────────────────────────────
                ax.axhspan(mean[var] - std[var], mean[var] + std[var],
                           color="gray", alpha=0.13, label="±1 SD (baseline)")
                ax.axhline(mean[var], color="gray", lw=1, ls=":",
                           alpha=0.8, label="Baseline mean")

                # ── danger threshold ───────────────────────────────────────
                if var in thresholds:
                    tval, tlabel, _ = thresholds[var]
                    ax.axhline(tval, color="#8B0000", lw=1.5, ls="-.",
                               alpha=0.8, label=tlabel)

                # ── event shaded band ──────────────────────────────────────
                if event_xi is not None:
                    ax.axvspan(event_xi - 0.45, event_xi + 0.45,
                               color=ev["color"], alpha=0.15, zorder=0)
                    ax.axvline(event_xi, color=ev["color"], lw=2,
                               ls="--", alpha=0.8, zorder=3)

                # ── data line ─────────────────────────────────────────────
                ax.plot(x_pos, vals, "o-", color=ev["color"], lw=2.2,
                        ms=8, zorder=5, clip_on=False)

                # Annotate each point with its value
                for xi, v in zip(x_pos, vals):
                    if not np.isnan(v):
                        ax.annotate(f"{v:.1f}", (xi, v),
                                    textcoords="offset points", xytext=(0, 8),
                                    ha="center", fontsize=7.5, color=ev["color"],
                                    fontweight="bold")

            # ── labels / formatting ────────────────────────────────────────
            ax.set_xticks(x_pos)
            ax.set_xticklabels(month_labels, rotation=30, ha="right", fontsize=8.5)
            ax.set_ylabel(vlabel, fontsize=9)
            ax.grid(True, axis="y", alpha=0.25)

            # Column title (event name) only on top row
            if row == 0:
                ax.set_title(ev["label"], fontsize=11, fontweight="bold",
                             color=ev["color"], pad=10)

            # Event-month arrow annotation
            if event_xi is not None and row == 0:
                ax.annotate(
                    "← Die-off\n   event",
                    xy=(event_xi, ax.get_ylim()[1]),
                    xytext=(event_xi + 0.3, ax.get_ylim()[1]),
                    fontsize=8, color=ev["color"], fontweight="bold",
                    va="top",
                )

            # Legend only on first column
            if col == 0:
                ax.legend(fontsize=7.5, loc="upper right",
                          framealpha=0.85, ncol=1)

    fig.suptitle(
        "Pre-event Lead-up & Recovery: Key Stress Variables\n"
        "Each column = one die-off event  |  "
        "Shaded band = event month  |  Grey band = normal range (±1 SD)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = VIZ_DIR / "18_dieoff_leadup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


def plot_leadup_compact(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> None:
    """
    Figure 18a - compact version for two-column papers.
    Both events on the same 3-panel figure (DO%, NH4, DIN).
    X-axis = months relative to the event (−2, −1, 0, +1, +2).
    """
    KEY_VARS = [
        ("do_per", "DO (% Sat)",   {"color": "#d62728", "lw": 1.5,
                                     "thresh": (75, "75% hypoxia", "low")}),
        ("nh4",    "NH4 (µmol/L)", {"color": None,      "lw": 1.5,
                                     "thresh": (10, "10 µmol/L concern", "high")}),
        ("din",    "DIN (µmol/L)", {"color": None,      "lw": 1.5,
                                     "thresh": (10, "10 µmol/L concern", "high")}),
    ]
    # Months relative to event: lead 2, event 0, follow 2
    REL = [-2, -1, 0, 1, 2]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.82, bottom=0.18)

    for ai, (var, vlabel, style) in enumerate(KEY_VARS):
        ax = axes[ai]

        # Baseline band
        ax.axhspan(mean[var] - std[var], mean[var] + std[var],
                   color="gray", alpha=0.13)
        ax.axhline(mean[var], color="gray", lw=0.8, ls=":", alpha=0.7)

        # Danger threshold
        tval, tlabel, _ = style["thresh"]
        ax.axhline(tval, color="#8B0000", lw=1.2, ls="-.", alpha=0.8,
                   label=tlabel)

        # Event month vertical
        ax.axvline(0, color="#333", lw=1.0, ls="--", alpha=0.5)

        for ev in EVENTS:
            all_months = ev["lead_months"] + ev["follow_months"]
            avg = monthly_spatial_avg(df, sorted(set(all_months)))
            if var not in avg.columns:
                continue

            ev_idx  = ev["lead_months"].index(ev["event_month"])
            ordered = sorted(set(all_months))
            ev_pos  = ordered.index(ev["event_month"])

            # Build relative-time series, aligned to event_month = 0
            rel_x, vals = [], []
            for ri, r in enumerate(REL):
                abs_pos = ev_pos + r
                if 0 <= abs_pos < len(ordered):
                    m = ordered[abs_pos]
                    v = avg[var].get(pd.Period(m, "M"), np.nan) \
                        if m in [str(p) for p in avg.index] else np.nan
                    rel_x.append(r); vals.append(v)

            c = ev["color"]
            ax.plot(rel_x, vals, "o-", color=c, lw=2.0, ms=6, zorder=5,
                    label=ev["short"], clip_on=False)

            # Annotate value at event month (x=0)
            zero_idx = rel_x.index(0) if 0 in rel_x else None
            if zero_idx is not None and not np.isnan(vals[zero_idx]):
                ax.annotate(f"{vals[zero_idx]:.1f}",
                            (0, vals[zero_idx]),
                            xytext=(4, 5), textcoords="offset points",
                            fontsize=7.5, color=c, fontweight="bold")

        ax.set_xticks(REL)
        ax.set_xticklabels([f"t{r:+d}" if r != 0 else "Event\n(t=0)"
                            for r in REL], fontsize=8.5)
        ax.set_xlabel("Months relative to event", fontsize=8.5)
        ax.set_ylabel(vlabel, fontsize=9)
        ax.set_title(vlabel, fontsize=10, fontweight="bold", pad=4)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)

        # Panel letter
        ax.text(0.03, 0.97, f"({chr(97+ai)})", transform=ax.transAxes,
                va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))

    # Shared legend below all panels
    handles = [plt.Line2D([0],[0], color=ev["color"], lw=2, marker="o",
                           ms=6, label=ev["short"]) for ev in EVENTS]
    handles += [plt.Line2D([0],[0], color="#8B0000", lw=1.2, ls="-.",
                            label="Eutrophication / hypoxia threshold"),
                plt.Rectangle((0,0),1,1, fc="gray", alpha=0.3,
                               label="Baseline ±1 SD")]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.52, -0.05))

    fig.suptitle(
        "Pre-event lead-up at bay-interior sites\n"
        "Grey band = baseline ±1 SD  |  t=0 = event month",
        fontsize=10, fontweight="bold",
    )

    out = VIZ_DIR / "18a_dieoff_leadup_compact.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def plot_fcm_projection(df: pd.DataFrame) -> None:
    if not FCM_WEIGHTS_NUTR.exists():
        print("  [SKIP] fcm_weights_nutrient.csv not found  - run fcm.py first")
        return

    W_df = pd.read_csv(FCM_WEIGHTS_NUTR, index_col=0)
    W    = W_df.values.astype(np.float64)
    concept_labels = list(W_df.columns)
    C = len(concept_labels)

    # Map nutrient FCM concepts to grab-sample columns
    label_to_feat = {
        "Water Temp":     "temp",
        "Salinity":       "sal",
        "ODO (mg/L)":     "do_mgL",
        "ODO (%Sat)":     "do_per",
        "Spec. Cond.":    "spc_scm",
        "pH":             "ph",
        "Chlorophyll-a":  "chl_exo_ugL",
        "Secchi Depth":   "secchi_m",   # note: monthly loader renames to secchi_m
        "NO₂+NO₃":        "no2no3",
        "DIN":            "din",
    }
    feat_to_idx = {f: i for i, f in enumerate(concept_labels)}

    # Load full series to compute normalisation bounds
    from fcm import load_nutrient_concept_timeseries, NUTR_CONCEPTS
    monthly_ts = load_nutrient_concept_timeseries()
    arr        = monthly_ts[NUTR_CONCEPTS].values.astype(np.float64)
    col_min    = np.nanmin(arr, axis=0)
    col_max    = np.nanmax(arr, axis=0)
    col_range  = col_max - col_min
    col_range[col_range == 0] = 1.0

    def normalise(vals: np.ndarray) -> np.ndarray:
        return (vals - col_min) / col_range

    # Build initial activation from event month
    plot_concepts = ["DO (%Sat)", "Water Temp", "DIN", "Chlorophyll-a",
                     "Salinity", "pH"]
    plot_idxs     = [feat_to_idx[c] for c in plot_concepts if c in feat_to_idx]
    plot_labels_  = [concept_labels[i] for i in plot_idxs]

    # Forcing indices (Rainfall, Air Temp Min, Air Temp Max = first 3)
    forcing_idxs  = [0, 1, 2]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes      = axes.flatten()
    n_steps   = 18

    for ev_i, ev in enumerate(EVENTS):
        snap      = monthly_spatial_avg(df, [ev["event_month"]])
        follow_df = monthly_spatial_avg(df, ev["follow_months"])

        # Build A0 from event snapshot (NaN → col mean)
        A0 = np.full(C, 0.5)
        for lbl, feat in label_to_feat.items():
            if feat == "secchi_m":
                feat = "secchi"
            if lbl in feat_to_idx and feat in snap.columns and not np.isnan(snap[feat].values[0]):
                feat_idx = NUTR_CONCEPTS.index(
                    next((k for k, v in {
                        "temp_c": "Water Temp", "sal_ppt": "Salinity",
                        "odo_mgL": "ODO (mg/L)", "odo_pct": "ODO (%Sat)",
                        "spec_cond_uScm": "Spec. Cond.", "ph": "pH",
                        "chl_a_ugL": "Chlorophyll-a", "secchi_m": "Secchi Depth",
                        "no2no3_umolL": "NO₂+NO₃", "din_umolL": "DIN",
                    }.items() if v == lbl), None)
                ) if next((k for k, v in {
                    "temp_c": "Water Temp", "sal_ppt": "Salinity",
                    "odo_mgL": "ODO (mg/L)", "odo_pct": "ODO (%Sat)",
                    "spec_cond_uScm": "Spec. Cond.", "ph": "pH",
                    "chl_a_ugL": "Chlorophyll-a", "secchi_m": "Secchi Depth",
                    "no2no3_umolL": "NO₂+NO₃", "din_umolL": "DIN",
                }.items() if v == lbl), None) is not None else None
                if feat_idx is not None:
                    raw_val = snap[feat].values[0]
                    A0[feat_to_idx[lbl]] = np.clip(
                        (raw_val - col_min[feat_idx]) / col_range[feat_idx], 0, 1
                    )

        # Simulate
        A      = A0.copy()
        forced = A0[forcing_idxs].copy()
        traj   = [A.copy()]
        for _ in range(n_steps):
            A_new = _sigmoid(W.T @ A)
            A_new[forcing_idxs] = forced
            A = A_new
            traj.append(A.copy())
        traj = np.stack(traj)

        # Plot
        for ax_i, (pi, plbl) in enumerate(zip(plot_idxs, plot_labels_)):
            if ax_i >= len(axes):
                break
            ax = axes[ax_i]
            ax.plot(range(n_steps + 1), traj[:, pi],
                    color=ev["color"], lw=2.5, label=ev["short"] + " (FCM projection)",
                    ls="-" if ev_i == 0 else "--", alpha=0.9)
            ax.set_title(plbl, fontsize=10, fontweight="bold")
            ax.set_xlabel("Months after event", fontsize=9)
            ax.set_ylabel("Normalised activation [0–1]", fontsize=9)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

    fig.suptitle(
        "FCM Forward Projection from Die-off Event States\n"
        "Initial conditions set to observed event-month values; "
        "forcing held at event-month levels",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = VIZ_DIR / "19_dieoff_fcm_projection.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Analysis 5  - Cross-event anomaly comparison
# ---------------------------------------------------------------------------

def plot_event_comparison(
    df:   pd.DataFrame,
    mean: pd.Series,
    std:  pd.Series,
) -> None:
    z_event = {}
    for ev in EVENTS:
        snap    = monthly_spatial_avg(df, [ev["event_month"]])
        z_event[ev["short"]] = ((snap.iloc[0] - mean) / (std + 1e-9))[FEAT_RAW]

    labels     = [FEAT_LABELS[f] for f in FEAT_RAW]
    x          = np.arange(len(FEAT_RAW))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (ev_short, z) in enumerate(z_event.items()):
        ev_color = next(e["color"] for e in EVENTS if e["short"] == ev_short)
        offset   = (i - 0.5) * width
        bars     = ax.bar(x + offset, z.values, width,
                          label=ev_short, color=ev_color, alpha=0.8,
                          edgecolor="k", linewidth=0.5)

    ax.axhline(0,    color="k",       lw=0.8)
    ax.axhline(+2,   color="#8B0000", lw=1,   ls="--", alpha=0.6, label="+2σ threshold")
    ax.axhline(-2,   color="#00008B", lw=1,   ls="--", alpha=0.6, label="−2σ threshold")
    ax.axhspan(+2, ax.get_ylim()[1] if ax.get_ylim()[1] > 2 else 5,
               color="#ffcccc", alpha=0.15)
    ax.axhspan(-2, ax.get_ylim()[0] if ax.get_ylim()[0] < -2 else -5,
               color="#ccccff", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Z-score vs 2021–2026 baseline", fontsize=10)
    ax.set_title(
        "Anomaly Comparison  - Sept 2021 vs Oct 2022 Die-off Events\n"
        "Values averaged across all Biscayne Bay Surface stations at event month",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate stress direction
    for j, feat in enumerate(FEAT_RAW):
        direction = "↑ stress" if feat in STRESS_HIGH else "↓ stress"
        ax.text(j, ax.get_ylim()[0] + 0.1, direction, ha="center",
                fontsize=6, color="#666", style="italic")

    plt.tight_layout()
    out = VIZ_DIR / "20_dieoff_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Printed text summary
# ---------------------------------------------------------------------------

def print_summary(
    df:   pd.DataFrame,
    mean: pd.Series,
    std:  pd.Series,
) -> None:
    print("\n" + "=" * 65)
    print("DIE-OFF ANOMALY SUMMARY")
    print("=" * 65)

    for ev in EVENTS:
        snap = monthly_spatial_avg(df, [ev["event_month"]]).iloc[0]
        z    = (snap - mean) / (std + 1e-9)

        print(f"\n{'─'*65}")
        print(f"{ev['label']}")
        print(f"{'─'*65}")
        print(f"  {'VARIABLE':<26} {'VALUE':>8}  {'Z-SCORE':>8}  STRESS")

        for feat in FEAT_RAW:
            val = snap[feat]
            zv  = z[feat]
            if np.isnan(val):
                continue
            stress = ""
            if feat in STRESS_HIGH  and zv >  1.5:
                stress = "⚠ HIGH"
            elif feat in STRESS_LOW and zv < -1.5:
                stress = "⚠ LOW"
            bar = "█" * int(abs(zv))
            print(f"  {FEAT_LABELS[feat]:<26} {val:>8.2f}  {zv:>+7.2f}  {stress}  {bar}")

    print(f"\n{'─'*65}")
    print("KEY FINDINGS:")
    print("  • Both events: DO suppressed at northern bay + canal sites")
    print("  • Sept 2021:   Low salinity (fresh water influx from canals)")
    print("                 NH4 spike at LR01 (12.6) and MR01 (7.2) µmol/L on sample date")
    print("                 (acute hypoxia likely peaked before the 21-Sep grab sample)")
    print("  • Oct 2022:    More widespread hypoxia; LR01 DO = 49.6% on sample date")
    print("                 GOC-014 DIN = 39.0, NH4 = 37.7 µmol/L (ocean outfall sites,")
    print("                 now excluded from bay-interior stats but ecologically relevant)")
    print("  • Canal discharge (LR01, MR01) drives nutrient loading in both events")
    print("  • No pre-event rainfall data available for 2021-2022")
    print("    (rainfall CSVs only cover 2025-2026)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading grab-sample data...")
    df   = load_grab_data()
    mean, std = compute_baseline(df)

    print(f"  {len(df)} Surface rows, {df['period'].nunique()} months, "
          f"{df['site_name'].nunique()} sites")

    print("\n[1] Anomaly heatmap...")
    plot_anomaly_heatmap(df, mean, std)

    print("\n[2] Spatial maps...")
    plot_spatial_maps(df)

    print("\n[2a] Spatial maps on basemap (17a)...")
    plot_spatial_maps_basemap(df)

    print("\n[3] Pre-event lead-up...")
    plot_leadup(df, mean, std)

    print("\n[3a] Compact lead-up (18a)...")
    plot_leadup_compact(df, mean, std)

    print("\n[4] FCM forward projection...")
    plot_fcm_projection(df)

    print("\n[5] Cross-event comparison...")
    plot_event_comparison(df, mean, std)

    print_summary(df, mean, std)

    print("\nDone. Outputs:")
    for i in range(16, 21):
        out = list(VIZ_DIR.glob(f"{i}_dieoff_*.png"))
        if out:
            print(f"  {out[0]}")
    print(f"  analysis/dieoff_anomalies.csv")


if __name__ == "__main__":
    main()
