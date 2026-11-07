"""
north_bay_hypoxia.py
--------------------
Observed-only dissolved-oxygen summary for the north-bay sensor nodes.

Reads imputed_output/*_imputed.csv but keeps only rows whose *_observed flag
is set, so no model-imputed value enters any statistic. Writes tables to
analysis/ and figures to visualizations/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IMPUTED_DIR = ROOT / "imputed_output"
ANALYSIS_DIR = ROOT / "analysis"
VIZ_DIR = ROOT / "visualizations"

# name -> (file stem, latitude N). The two southern nodes are reference sites.
# Source change: from Jun 2026 the L0, L2, L7, p2 and crest5 nodes also take
# buoy data (L0=Haulover Inlet, L2=CREST3 ~1.9 km away, L7=CREST4, p2=Royal
# Castle, crest5=CREST5+Kampong). Only L1 (p4) and L6 (p8) keep the same sensor.
NODES = {
    "L0": ("raw-data-platformL0_parameters", 25.911),
    "L1 canal": ("raw-data-platformL1_parameters", 25.875),
    "p2 bay": ("biscayne_bay", 25.870),
    "L2": ("raw-data-platformL2_parameters", 25.854),
    "L6": ("raw-data-platformL6_parameters", 25.844),
    "L7 (south)": ("raw-data-platformL7_parameters", 25.779),
    "crest5 (south)": ("consolidated_crest5", 25.727),
}
SOURCE_NOTE = ("Jun 2026 onward, L0/L2/L7/p2/crest5 also include buoy data "
               "(Haulover, CREST3, CREST4, Royal Castle, CREST5+Kampong); only L1 and L6 keep the same sensor.")
SAMPLES_PER_DAY = 288
MIN_DAY_COVERAGE = 0.80
HYPOXIA = 4.0
SEVERE = 2.0
FLATLINE_SAMPLES = 12  # one hour at 5-minute cadence


def load_observed(stem: str) -> pd.DataFrame:
    df = pd.read_csv(IMPUTED_DIR / f"{stem}_imputed.csv", index_col=0, parse_dates=True)
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)
    keep = df["odo_mgL_observed"].astype(bool)
    return df.loc[keep, ["odo_mgL", "temp_c", "sal_ppt"]]


def flag_suspect(odo: pd.Series) -> pd.Series:
    """Exact zeros or an hour of identical readings suggest a sensor fault."""
    flat = odo.rolling(FLATLINE_SAMPLES).std().eq(0)
    flat = flat.rolling(FLATLINE_SAMPLES, min_periods=1).max().astype(bool)
    return (odo <= 0) | flat


def daily_summary(name: str, obs: pd.DataFrame) -> pd.DataFrame:
    suspect = flag_suspect(obs["odo_mgL"])
    day = obs.index.floor("D")
    g = obs.assign(
        low=(obs["odo_mgL"] < HYPOXIA),
        severe=(obs["odo_mgL"] < SEVERE),
        suspect=suspect,
    ).groupby(day)
    out = pd.DataFrame(
        {
            "coverage": g["odo_mgL"].size() / SAMPLES_PER_DAY,
            "odo_mean": g["odo_mgL"].mean(),
            "odo_min": g["odo_mgL"].min(),
            "odo_range": g["odo_mgL"].max() - g["odo_mgL"].min(),
            "hours_lt4": g["low"].sum() * 24 / SAMPLES_PER_DAY,
            "hours_lt2": g["severe"].sum() * 24 / SAMPLES_PER_DAY,
            "suspect_frac": g["suspect"].mean(),
            "temp_mean": g["temp_c"].mean(),
            "sal_mean": g["sal_ppt"].mean(),
        }
    )
    out = out[out["coverage"] >= MIN_DAY_COVERAGE].copy()
    out.insert(0, "node", name)
    out.index.name = "date"
    return out


def plot_heatmap(daily: pd.DataFrame, path: Path) -> None:
    pivot = daily.pivot_table(index="node", columns="date", values="hours_lt4")
    full = pd.date_range(pivot.columns.min(), pivot.columns.max(), freq="D")
    pivot = pivot.reindex(index=list(NODES), columns=full)
    fig, ax = plt.subplots(figsize=(15, 4.6))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(
        np.ma.masked_invalid(pivot.values),
        aspect="auto", cmap=cmap, vmin=0, vmax=24,
        extent=[0, len(full), len(pivot) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ticks = [i for i, d in enumerate(full) if d.day == 1 and d.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([full[i].strftime("%b %Y") for i in ticks])
    ax.set_title("Hours per day with dissolved oxygen below 4 mg/L (observed days only; grey = no data)")
    fig.colorbar(im, ax=ax, label="hours / day")
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_year_compare(daily: pd.DataFrame, path: Path) -> None:
    d = daily.copy()
    d["year"] = d.index.year
    d["month"] = d.index.month
    names = ["L0", "L1 canal", "L2"]
    fig, axes = plt.subplots(1, len(names), figsize=(13, 3.8), sharey=True)
    for ax, name in zip(axes, names):
        for year, color in ((2025, "#4c78a8"), (2026, "#e45756")):
            s = d[(d["node"] == name) & (d["year"] == year)]
            m = s.groupby("month")["hours_lt4"].mean().reindex(range(1, 13))
            ax.plot(m.index, m.values, marker="o", color=color, label=str(year))
        ax.set_title(name if name == "L1 canal" else f"{name} (sensor changes Jun 2026)")
        ax.set_xlabel("month")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel("mean hours/day below 4 mg/L")
    axes[0].legend()
    fig.suptitle("Same node, same month: 2025 vs 2026 (days with >=80% sensor coverage). "
                 "Only L1 is like-for-like.")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    frames = [daily_summary(name, load_observed(stem)) for name, (stem, _) in NODES.items()]
    daily = pd.concat(frames)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    daily.to_csv(ANALYSIS_DIR / "north_bay_hypoxia_daily.csv", float_format="%.4f")

    monthly = (
        daily.groupby(["node", daily.index.to_period("M")])
        .agg(
            days=("odo_mean", "size"),
            odo_mean=("odo_mean", "mean"),
            odo_min=("odo_min", "min"),
            hours_lt4_per_day=("hours_lt4", "mean"),
            hours_lt2_per_day=("hours_lt2", "mean"),
            days_with_lt2=("hours_lt2", lambda x: int((x > 0).sum())),
            suspect_frac=("suspect_frac", "mean"),
        )
        .reset_index()
    )
    monthly = monthly[monthly["days"] >= 10]
    monthly["node"] = pd.Categorical(monthly["node"], list(NODES), ordered=True)
    monthly = monthly.sort_values(["node", "date"])
    monthly.to_csv(ANALYSIS_DIR / "north_bay_hypoxia_monthly.csv", index=False, float_format="%.3f")

    plot_heatmap(daily, VIZ_DIR / "30_north_bay_hypoxia_heatmap.png")
    plot_year_compare(daily, VIZ_DIR / "31_north_bay_hypoxia_2025_vs_2026.png")

    print(f"daily rows: {len(daily)}  nodes: {daily['node'].nunique()}")
    print("suspect fraction (zero or flat-line readings) by node:")
    print(daily.groupby("node")["suspect_frac"].mean().round(4).to_string())
    recent = monthly[monthly["date"] >= pd.Period("2026-06")]
    print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
