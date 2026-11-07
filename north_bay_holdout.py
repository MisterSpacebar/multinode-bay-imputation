"""
north_bay_holdout.py
--------------------
Chronological hold-out test for forecasting daily hypoxia hours (DO < 4 mg/L)
at north-bay nodes, using observed-only daily summaries from
north_bay_hypoxia.py (analysis/north_bay_hypoxia_daily.csv).

For each test month the models are fit only on days whose target date is
before that month starts. Forecasts are compared with two baselines:
  persistence : hours today
  trail7      : mean of the last 7 days
and a ridge model (fixed alpha, no tuning on test data) using the node's own
daily state plus rainfall/temperature forcing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DAILY = ROOT / "analysis" / "north_bay_hypoxia_daily.csv"
OUT = ROOT / "analysis" / "north_bay_holdout_results.csv"

NODES = ["L1 canal", "L2", "p2 bay"]
TESTS = [("2025-09", "2025-09-01", "2025-09-30"), ("2026-09", "2026-09-01", "2026-09-30")]
HORIZONS = [1, 3, 7]
ALPHA = 10.0
FEATURES = ["hours_lt4", "odo_mean", "odo_min", "odo_range", "temp_mean", "sal_mean",
            "trail7", "net_water", "temp_max"]


def load_forcing() -> pd.DataFrame:
    import contextlib
    import io

    import fcm

    with contextlib.redirect_stdout(io.StringIO()):
        phys = fcm.load_physical_concept_timeseries()
    return phys[["net_water", "temp_max"]]


def node_frame(daily: pd.DataFrame, forcing: pd.DataFrame, node: str) -> pd.DataFrame:
    d = daily[daily["node"] == node].set_index("date")
    d.index = pd.to_datetime(d.index)
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d["trail7"] = d["hours_lt4"].rolling(7, min_periods=4).mean()
    return d.join(forcing, how="left")


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float):
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - ym))
    return lambda q: np.clip(((q - mu) / sd) @ w + ym, 0, 24)


def main() -> None:
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    forcing = load_forcing()
    rows = []
    for node in NODES:
        d = node_frame(daily, forcing, node)
        for h in HORIZONS:
            d["target"] = d["hours_lt4"].shift(-h)
            for label, start, end in TESTS:
                s, e = pd.Timestamp(start), pd.Timestamp(end)
                target_date = d.index + pd.Timedelta(days=h)
                train = d[(target_date < s) & d["target"].notna()].dropna(subset=FEATURES)
                test = d[(target_date >= s) & (target_date <= e) & d["target"].notna()]
                test = test.dropna(subset=FEATURES)
                if len(train) < 40 or len(test) < 10:
                    continue
                predict = ridge_fit(train[FEATURES].values, train["target"].values, ALPHA)
                y = test["target"].values
                preds = {
                    "persistence": test["hours_lt4"].values,
                    "trail7": test["trail7"].values,
                    "ridge": predict(test[FEATURES].values),
                    "train_mean": np.full(len(test), train["target"].mean()),
                }
                mse_p = np.mean((y - preds["persistence"]) ** 2)
                for name, p in preds.items():
                    rows.append({
                        "node": node, "test_month": label, "horizon_d": h,
                        "n_train": len(train), "n_test": len(test), "model": name,
                        "mae_h": np.mean(np.abs(y - p)),
                        "skill_vs_persist": 1 - np.mean((y - p) ** 2) / mse_p if mse_p > 0 else np.nan,
                        "obs_mean_h": y.mean(),
                    })
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    wide = res.pivot_table(index=["node", "test_month", "horizon_d", "n_train", "n_test", "obs_mean_h"],
                           columns="model", values="mae_h").round(2)
    print("MAE (hours/day), lower is better")
    print(wide.to_string())


if __name__ == "__main__":
    main()
