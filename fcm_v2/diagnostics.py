"""Diagnostics for observed-only FCM v2 forecasts and learned edges.

Read existing local exports, not .url Windows shortcuts. No private data
or GPU is needed. Not a model training script.

Run from repository root:
    python -m fcm_v2.diagnostics
    python -m fcm_v2.diagnostics --bootstrap 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("node", "issue_date", "target_date", "horizon_days")
COMPARE = ("fcm_linear_physical", "fcm_linear_do", "ridge_physical", "persistence")


def load_predictions(fcm: pd.DataFrame, baselines: pd.DataFrame,
                     models=COMPARE) -> pd.DataFrame:
    """Select requested models and align by node, origin and target date.

    Actual observations must agree across sources. Fail rather than compare
    mismatched labels or duplicated forecast rows.
    """
    required = set(KEYS) | {"model", "actual_hours_lt4", "predicted_hours_lt4"}
    for name, frame in (("FCM", fcm), ("baseline", baselines)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")
    requested = set(models)
    f = pd.concat([fcm, baselines], ignore_index=True)
    f = f.loc[f["model"].isin(requested), list(KEYS) + [
        "model", "actual_hours_lt4", "predicted_hours_lt4"
    ]].copy()
    if f.empty:
        raise ValueError("No requested models exist in the input predictions")
    if f.duplicated(list(KEYS) + ["model"]).any():
        raise ValueError("Duplicate model/issue/target keys; check source files")
    f["issue_date"] = pd.to_datetime(f["issue_date"])
    f["target_date"] = pd.to_datetime(f["target_date"])
    f["actual_hours_lt4"] = pd.to_numeric(
        f["actual_hours_lt4"], errors="raise")
    f["predicted_hours_lt4"] = pd.to_numeric(
        f["predicted_hours_lt4"], errors="raise")
    if f[["actual_hours_lt4", "predicted_hours_lt4"]].isna().any().any():
        raise ValueError("Predictions or measured targets contain NaN")
    if not (f["issue_date"] < f["target_date"]).all():
        raise ValueError("Forecast target must be strictly later than issue")
    labels = f.groupby(list(KEYS))["actual_hours_lt4"]
    if labels.agg(lambda x: float(x.max() - x.min())).gt(1e-6).any():
        raise ValueError("Target observation differs across aligned models")
    return f


def _paired(f: pd.DataFrame, candidate: str, reference: str) -> pd.DataFrame:
    a = f[f["model"].eq(candidate)].drop(columns="model").rename(
        columns={"predicted_hours_lt4": "a",
                 "actual_hours_lt4": "actual_a"})
    b = f[f["model"].eq(reference)].drop(columns="model").rename(
        columns={"predicted_hours_lt4": "b",
                 "actual_hours_lt4": "actual_b"})
    m = a.merge(b, on=list(KEYS), how="inner", validate="one_to_one")
    if m.empty:
        return m
    if not np.allclose(m["actual_a"], m["actual_b"], atol=1e-6, rtol=0):
        raise ValueError("Paired models do not have matching actual DO labels")
    m["y"] = m["actual_a"]
    m["err_sq_a"] = (m["y"] - m["a"]) ** 2
    m["err_sq_b"] = (m["y"] - m["b"]) ** 2
    m["month"] = m["target_date"].dt.to_period("M").astype(str)
    return m


def _rmse_pair(m: pd.DataFrame):
    a, b = m["err_sq_a"], m["err_sq_b"]
    return float(np.sqrt(a.mean())), float(np.sqrt(b.mean()))


def _bootstrap_rmse_difference(m: pd.DataFrame, n_boot: int,
                               seed: int) -> tuple[float, float]:
    """Station-stratified *station-month* cluster bootstrap.

    Resamples calendar-month blocks WITHIN each station, retaining each
    station's original number of blocks and individual days per sampled
    month. This only approximately captures dependence and should not be
    interpreted as a definitive hypothesis test.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    rng = np.random.default_rng(seed)
    clusters = {}
    for (node, month), part in m.groupby(["node", "month"], sort=True):
        clusters.setdefault(node, []).append(np.array([
            float(part["err_sq_a"].sum()),
            float(part["err_sq_b"].sum()),
            float(len(part)),
        ]))
    blocks = {key: np.stack(v, axis=0) for key, v in clusters.items()}
    draws = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        totals = np.zeros(3, dtype=float)
        for v in blocks.values():
            choices = rng.integers(0, len(v), size=len(v))
            totals += v[choices].sum(axis=0)
        draws[iteration] = (
            np.sqrt(totals[0] / totals[2])
            - np.sqrt(totals[1] / totals[2])
        )
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def compare_models(pred: pd.DataFrame, n_boot: int = 2000,
                   seed: int = 7,
                   pairs=(("fcm_linear_physical", "persistence"),
                          ("fcm_linear_physical", "fcm_linear_do"),
                          ("fcm_linear_physical", "ridge_physical"))) -> pd.DataFrame:
    """Matched case comparison and exploratory clustered uncertainty.

    Negative difference = candidate wins. Tests are NOT multiplicity-adjusted.
    """
    rows = []
    for horizon in sorted(pred["horizon_days"].unique()):
        p = pred[pred["horizon_days"].eq(horizon)]
        for a, b in pairs:
            m = _paired(p, a, b)
            if m.empty:
                continue
            ra, rb = _rmse_pair(m)
            low, high = _bootstrap_rmse_difference(
                m, n_boot=n_boot, seed=seed + int(horizon))
            rows.append({
                "horizon_days": int(horizon),
                "candidate": a,
                "reference": b,
                "n_cases": len(m),
                "n_stations": m["node"].nunique(),
                "n_station_month_blocks": m.groupby(["node", "month"]).ngroups,
                "n_hypoxic_target_days": int((m["y"] > 0).sum()),
                "candidate_rmse_hours": ra,
                "reference_rmse_hours": rb,
                "delta_rmse_hours": ra - rb,
                "delta_rmse_ci_low_approx": low,
                "delta_rmse_ci_high_approx": high,
                "candidate_mae_hypoxic_days": (
                    float(np.abs(m.loc[m["y"] > 0, "y"] -
                                 m.loc[m["y"] > 0, "a"]).mean())
                    if (m["y"] > 0).any() else np.nan),
                "reference_mae_hypoxic_days": (
                    float(np.abs(m.loc[m["y"] > 0, "y"] -
                                 m.loc[m["y"] > 0, "b"]).mean())
                    if (m["y"] > 0).any() else np.nan),
                "ci_method": "station_stratified_station_month_bootstrap_approx",
            })
    return pd.DataFrame(rows)


STABILITY_COLUMNS = (
    "node", "horizon_days", "activation", "concept_set", "source",
    "target", "n_origins", "n_origin_months", "median_weight",
    "mean_weight", "p10_weight", "p90_weight", "share_near_zero",
    "sign_consistency_nonzero", "n_significant_sign_flips",
    "early_median", "late_median", "median_shift",
    "stable_exploratory",
)


def edge_stability(edges: pd.DataFrame, near_zero=0.05,
                   min_origins=20) -> pd.DataFrame:
    """Track coefficient sign and magnitude changes across forecast origins.

    Frequent refits share much of the same training data. They are heavily
    dependent and are NOT independent replications or causal evidence.
    """
    cols = ("node", "horizon_days", "activation", "concept_set",
            "source", "target", "asof_date", "weight")
    if not set(cols).issubset(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(set(cols) - set(edges))}")
    e = edges[list(cols)].copy()
    e["asof_date"] = pd.to_datetime(e["asof_date"], errors="raise")
    e["weight"] = pd.to_numeric(e["weight"], errors="coerce")
    if e.duplicated(list(cols[:-2]) + ["asof_date"]).any():
        raise ValueError("Duplicate fitted edge for same origin/model")
    rows = []
    group_cols = list(cols[:6])
    for key, grp in e.groupby(group_cols, sort=True):
        grp = grp.sort_values("asof_date")
        w = grp["weight"].to_numpy(dtype=float)
        finite = np.isfinite(w)
        w = w[finite]
        ts = grp.loc[finite, "asof_date"]
        if not len(w):
            continue
        meaningful = w[np.abs(w) > near_zero]
        dominant = (max((meaningful > 0).sum(), (meaningful < 0).sum())
                    / len(meaningful) if len(meaningful) else np.nan)
        significant_signs = np.sign(meaningful)
        flips = int((significant_signs[1:] != significant_signs[:-1]).sum())
        half = len(w) // 2
        early = float(np.median(w[:half])) if half else np.nan
        late = float(np.median(w[half:]))
        median = float(np.median(w))
        share_weak = float(np.mean(np.abs(w) <= near_zero))
        rows.append(dict(zip(group_cols, key)) | {
            "n_origins": len(w),
            "n_origin_months": int(ts.dt.to_period("M").nunique()),
            "median_weight": median,
            "mean_weight": float(np.mean(w)),
            "p10_weight": float(np.quantile(w, 0.10)),
            "p90_weight": float(np.quantile(w, 0.90)),
            "share_near_zero": share_weak,
            "sign_consistency_nonzero": float(dominant),
            "n_significant_sign_flips": flips,
            "early_median": early, "late_median": late,
            "median_shift": late - early if np.isfinite(early) else np.nan,
            "stable_exploratory": bool(
                len(w) >= min_origins
                and ts.dt.to_period("M").nunique() >= 3
                and len(meaningful) >= min_origins // 2
                and dominant >= 0.8 and flips <= 3
                and np.sign(early) == np.sign(late)
            ),
        })
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def summarize_stable_edges(stability: pd.DataFrame) -> pd.DataFrame:
    """Across-station counts of exploratory stability for incoming DO links."""
    if stability.empty:
        return stability.copy()
    d = stability.loc[
        stability["target"].eq("hours_lt4") &
        stability["source"].ne("hours_lt4") &
        stability["concept_set"].eq("physical")
    ]
    if d.empty:
        return d.copy()
    return d.groupby(
        ["horizon_days", "activation", "source", "target"],
        as_index=False,
    ).agg(
        n_station_series=("node", "nunique"),
        n_stable_series=("stable_exploratory", "sum"),
        n_positive_median=("median_weight", lambda w: int((w > 0).sum())),
        n_negative_median=("median_weight", lambda w: int((w < 0).sum())),
        median_of_station_medians=("median_weight", "median"),
        median_sign_consistency=("sign_consistency_nonzero", "median"),
    )


def main():
    parser = argparse.ArgumentParser(description="FCM edge/forecast diagnostics")
    parser.add_argument("--edges", type=Path,
                        default=Path("fcm_v2_outputs/learned_edges.csv"))
    parser.add_argument("--fcm", type=Path,
                        default=Path("fcm_v2_outputs/predictions.csv"))
    parser.add_argument("--baselines", type=Path,
                        default=Path("trustworthy_outputs/backtest_predictions.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("fcm_v2_outputs/diagnostics"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--near-zero", type=float, default=0.05)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.edges.exists():
        parser.error(f"No real edge CSV at {args.edges}")
    edges = pd.read_csv(args.edges)
    stability = edge_stability(edges, near_zero=args.near_zero)
    stability.to_csv(args.out_dir / "edge_stability.csv", index=False)
    by_source = summarize_stable_edges(stability)
    by_source.to_csv(args.out_dir / "edge_do_summary.csv", index=False)
    print(f"Edge series: {len(stability)}; candidate stable: "
          f"{int(stability['stable_exploratory'].sum())}")
    if args.fcm.exists() and args.baselines.exists():
        fcm = pd.read_csv(args.fcm)
        baselines = pd.read_csv(args.baselines)
        data = load_predictions(fcm, baselines)
        comparisons = compare_models(data, args.bootstrap, args.seed)
        comparisons.to_csv(args.out_dir / "paired_uncertainty.csv",
                           index=False)
        print(comparisons.to_string(index=False))
    else:
        print("Forecast CSV absent. Edge stability generated; paired "
              "uncertainty skipped.")
    print(f"Saved diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
