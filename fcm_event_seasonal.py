"""Seasonally matched die-off anomalies (graphic 06a).

Each event month is compared with the same calendar month in the other
sampled years, instead of the pooled multi-year baseline used in graphic 06.
Run: python fcm_event_seasonal.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dieoff_analysis as da
from fcm_presentation import NEGATIVE, ROOT, save_figure

FEATURES = ["temp", "do_mgL", "do_per", "ph", "din"]
LABELS = ["Water temperature", "Dissolved oxygen (mg/L)", "Oxygen saturation", "pH", "DIN"]
EVENTS = [(2021, 9, "#245b96"), (2022, 10, "#ac6c00")]


def seasonal_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, month, _ in EVENTS:
        in_month = df[df["datetime"].dt.month == month]
        event = in_month[in_month["datetime"].dt.year == year]
        base = in_month[in_month["datetime"].dt.year != year]
        base_years = sorted(base["datetime"].dt.year.unique())
        for feat in FEATURES:
            rows.append({
                "event": f"{year}-{month:02d}",
                "feature": feat,
                "event_mean": event[feat].mean(),
                "same_month_mean": base[feat].mean(),
                "same_month_std": base[feat].std(),
                "seasonal_z": (event[feat].mean() - base[feat].mean()) / base[feat].std(),
                "n_event_samples": int(event[feat].notna().sum()),
                "n_baseline_samples": int(base[feat].notna().sum()),
                "baseline_years": ",".join(str(y) for y in base_years),
            })
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, pooled: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.20, top=0.83)
    y = np.arange(len(FEATURES))
    for offset, (year, month, color) in zip((-0.18, 0.18), EVENTS):
        key = f"{year}-{month:02d}"
        sub = table[table["event"] == key].set_index("feature").loc[FEATURES]
        values = sub["seasonal_z"].to_numpy(dtype=float)
        ax.barh(y + offset, values, height=0.32, color=color,
                label=f"{key} vs same month in {sub['baseline_years'].iloc[0].replace(',', ', ')}")
        pooled_values = pooled.loc[key, FEATURES].to_numpy(dtype=float)
        ax.scatter(pooled_values, y + offset,
                   marker="D", s=70, facecolor="white", edgecolor="#243443", zorder=3)
        for i, value in enumerate(values):
            # Place the label beyond both the bar and its pooled marker so they never overlap.
            if value >= 0:
                ax.text(max(value, pooled_values[i]) + 0.1, i + offset, f"{value:+.2f}",
                        ha="left", va="center", fontsize=12)
            else:
                ax.text(min(value, pooled_values[i]) - 0.1, i + offset, f"{value:+.2f}",
                        ha="right", va="center", fontsize=12)
    ax.scatter([], [], marker="D", s=70, facecolor="white", edgecolor="#243443",
               label="Pooled anomaly from graphic 06 (not season-matched)")
    ax.set_yticks(y, LABELS, fontsize=13)
    ax.set_xlim(-3.0, 2.0)
    ax.axvline(0, color="#243443", lw=1)
    ax.set_xlabel("Standardized anomaly vs the same calendar month in other years", fontsize=12)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    fig.suptitle("Event months vs the same month in other years", fontsize=23, weight="bold")
    fig.text(0.5, 0.89, "Descriptive measurements, NOT FCM predictions or causal attribution.",
             ha="center", fontsize=14, color=NEGATIVE)
    fig.text(0.5, 0.05,
             "Baselines are only 2-3 other years (Sep 2024 was not sampled), so the reference is itself noisy.\n"
             "Anomaly = event-month site mean minus same-month mean of other years, divided by that month's sample std.",
             ha="center", fontsize=12)
    save_figure(fig, "06a_event_context_seasonal")


def main() -> None:
    df = da.load_grab_data()
    table = seasonal_anomalies(df)
    table.to_csv(ROOT / "analysis" / "fcm_event_seasonal.csv", index=False, float_format="%.3f")
    pooled = pd.read_csv(ROOT / "analysis" / "dieoff_anomalies.csv", index_col=0)
    plot(table, pooled)
    print(table.pivot(index="feature", columns="event", values="seasonal_z").loc[FEATURES].round(2))
    print(table[["event", "n_event_samples", "n_baseline_samples", "baseline_years"]].drop_duplicates())


if __name__ == "__main__":
    main()
