"""
generate_report_v2.py
---------------------
Report v2  - every chart/graph replaced by the underlying data table.
Writes report/figures_summary_v2.txt

Run:
    python generate_report_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ANALYSIS_DIR = Path("analysis")
IMPUTED_DIR  = Path("imputed_output")
GRAB_CSV     = Path("water_data/2021_to_2024/BB_thru2024.csv")
OUT_DIR      = Path("report")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────

def tbl(df, float_fmt=".3f", indent=2):
    """Format a DataFrame as a fixed-width ASCII table string."""
    pad = " " * indent
    col_widths = {}
    for c in df.columns:
        vals = [str(c)] + [
            (f"{v:{float_fmt}}" if isinstance(v, (float, np.floating)) else str(v))
            for v in df[c]
        ]
        col_widths[c] = max(len(v) for v in vals)
    header = pad + "  ".join(str(c).ljust(col_widths[c]) for c in df.columns)
    sep    = pad + "  ".join("-" * col_widths[c] for c in df.columns)
    rows   = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = f"{v:{float_fmt}}"
            else:
                s = str(v)
            cells.append(s.ljust(col_widths[c]))
        rows.append(pad + "  ".join(cells))
    return "\n".join(rows)

def matrix_tbl(df, float_fmt=".3f", indent=2):
    """Format a square matrix DataFrame with row-index labels."""
    pad = " " * indent
    idx_w = max(len(str(i)) for i in df.index)
    col_w = max(
        max(len(str(c)) for c in df.columns),
        len(f"{0:{float_fmt}}")
    )
    header = pad + " " * (idx_w + 2) + "  ".join(str(c)[:col_w].ljust(col_w) for c in df.columns)
    sep    = pad + "-" * (idx_w + 2 + (col_w + 2) * len(df.columns))
    rows   = [header, sep]
    for idx, row in df.iterrows():
        cells = [str(idx).ljust(idx_w)]
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = f"{v:{float_fmt}}"
            else:
                s = str(v)
            cells.append(s.ljust(col_w))
        rows.append(pad + "  ".join(cells))
    return "\n".join(rows)

def h1(t): return "\n" + "=" * 72 + f"\n{t.upper()}\n" + "=" * 72
def h2(t): return "\n" + "-" * 60 + f"\n{t}\n" + "-" * 60
def p(t=""):  return t

# ── load data ───────────────────────────────────────────────────────────────

ch_imp  = pd.read_csv(ANALYSIS_DIR / "channel_importance.csv")
cross   = pd.read_csv(ANALYSIS_DIR / "cross_feature_matrix.csv", index_col=0)
sp_imp  = pd.read_csv(ANALYSIS_DIR / "spatial_importance.csv",  index_col=0)
attn    = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv",   index_col=0)
W_phys  = pd.read_csv(ANALYSIS_DIR / "fcm_weights_physical.csv",  index_col=0)
W_nutr  = pd.read_csv(ANALYSIS_DIR / "fcm_weights_nutrient.csv",  index_col=0)
hm      = pd.read_csv(ANALYSIS_DIR / "fcm_hindcast_metrics.csv")
da      = pd.read_csv(ANALYSIS_DIR / "dieoff_anomalies.csv")

# ── load imputed node data for map tables ───────────────────────────────────

IMPUTED_FILES = {
    "L0":                  "raw-data-platformL0_parameters_imputed.csv",
    "L1":                  "raw-data-platformL1_parameters_imputed.csv",
    "L2":                  "raw-data-platformL2_parameters_imputed.csv",
    "L6":                  "raw-data-platformL6_parameters_imputed.csv",
    "L7":                  "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay":        "biscayne_bay_imputed.csv",
    "consolidated_crest5": "consolidated_crest5_imputed.csv",
}
SEASONS = {"Win(DJF)": [12,1,2], "Spr(MAM)": [3,4,5],
           "Sum(JJA)": [6,7,8],  "Aut(SON)": [9,10,11]}

def load_sensor_seasonal(feat):
    """Annual × seasonal median for one feature, observed rows only."""
    records = []
    for name, fname in IMPUTED_FILES.items():
        df = pd.read_csv(IMPUTED_DIR / fname, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        if feat not in df.columns:
            continue
        obs_col = feat + "_observed"
        if obs_col in df.columns:
            df.loc[df[obs_col] == 0, feat] = np.nan
        df["year"]   = df.index.year
        df["month"]  = df.index.month
        m2s = {m: s for s, ms in SEASONS.items() for m in ms}
        df["season"] = df["month"].map(m2s)
        df.loc[df["month"] == 12, "year"] += 1
        for (yr, ss), grp in df.groupby(["year","season"]):
            v = grp[feat].median()
            if not np.isnan(v):
                records.append({"station": name, "year": yr, "season": ss,
                                 "value": round(v, 2)})
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    pivot = out.pivot_table(index="station", columns=["year","season"],
                             values="value", aggfunc="median")
    pivot.columns = [f"{y} {s}" for y, s in pivot.columns]
    return pivot.reset_index()

def load_grab_seasonal(var, site_types=None):
    """Annual × seasonal median for one grab-sample variable, surface only."""
    df = pd.read_csv(GRAB_CSV, encoding="latin-1")
    df = df[df["sample_type"] == "Surface"].copy()
    df = df[~df["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()
    if site_types:
        df = df[df["site_type"].isin(site_types)]
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["site_name"] = df["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)
    m2s = {m: s for s, ms in SEASONS.items() for m in ms}
    df["season"] = df["month"].map(m2s)
    df.loc[df["month"] == 12, "year"] += 1
    df[var] = pd.to_numeric(df[var], errors="coerce")
    pivot = df.pivot_table(index="site_name", columns=["year","season"],
                            values=var, aggfunc="median")
    pivot.columns = [f"{y} {s}" for y, s in pivot.columns]
    return pivot.reset_index()

# ── build FCM edge table ─────────────────────────────────────────────────────

def fcm_edge_table(W_df, top_n=20):
    rows = []
    for src in W_df.index:
        for tgt in W_df.columns:
            if src == tgt:
                continue
            w = float(W_df.loc[src, tgt])
            if abs(w) > 0.05:
                rows.append({"Source": src, "Direction": "→" if w > 0 else "⊣",
                              "Target": tgt, "Weight": round(w, 3),
                              "Effect": "promotes" if w > 0 else "suppresses"})
    rows.sort(key=lambda x: abs(x["Weight"]), reverse=True)
    return pd.DataFrame(rows[:top_n])

# ── build report ────────────────────────────────────────────────────────────

blocks = []
A = blocks.append

A("BISCAYNE BAY WATER QUALITY ANALYSIS  - FIGURE SUMMARY v2")
A(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
A("Tables show the underlying data for every chart and figure.")
A("=" * 72)

# ============================================================
A(h1("SECTION 1  - ST-GNN IMPUTATION RESULTS (Figures 01–07)"))
# ============================================================

A(h2("Figure 01  - Time-series overview"))
A("Data: hourly-resampled observed + imputed values for temp_c, sal_ppt,")
A("odo_mgL at all 7 nodes. Imputed regions are shaded orange in the chart.")
A("No single table  - see Figure 06 per-node detail tables below (Sec 1).")

A(h2("Figure 02  - Observed vs imputed coverage (% missing per node)"))
A("Source: build_dataset() output at last run.\n")
cov_data = {
    "Node":           ["L0","L1","L2","L6","L7","biscayne_bay","consolidated_crest5"],
    "Missing (%)":    [68.6, 52.2, 39.6, 89.4, 85.0, 84.8, 92.5],
    "Observed (%)":   [31.4, 47.8, 60.4, 10.6, 15.0, 15.2,  7.5],
}
A(tbl(pd.DataFrame(cov_data), float_fmt=".1f"))

A(h2("Figure 03  - Channel permutation importance"))
A("Metric: delta loss = (loss with feature masked) - (baseline loss).")
A("Negative = masking that feature hurts reconstruction (feature is useful).")
A("Source: analysis/channel_importance.csv\n")
ci_disp = ch_imp[["feature","baseline_loss","masked_loss","importance"]].copy()
ci_disp = ci_disp.sort_values("importance").reset_index(drop=True)
ci_disp.columns = ["Feature","Baseline Loss","Masked Loss","Delta (Importance)"]
A(tbl(ci_disp, float_fmt=".6f"))

A(h2("Figure 03b  - Cross-feature dependency matrix"))
A("Entry [row, col]: extra loss on 'col' feature when 'row' is also masked.")
A("Positive = row feature helps predict col feature.")
A("Source: analysis/cross_feature_matrix.csv\n")
A(matrix_tbl(cross.round(5), float_fmt=".5f"))

A(h2("Figure 04  - Spatial graph (top edge importances)"))
A("Edge importance = increase in reconstruction loss when that directed")
A("edge is removed from the graph attention layer.")
A("Source: analysis/spatial_importance.csv\n")
sp_rows = []
sp_T = sp_imp.T  # rows=source, cols=target
for src in sp_T.index:
    for tgt in sp_T.columns:
        try:
            v = float(sp_T.loc[src, tgt])
        except Exception:
            continue
        if abs(v) > 1e-9:
            sp_rows.append({"Source": src, "Target": tgt,
                             "Importance": round(v, 9)})
sp_rows.sort(key=lambda x: abs(x["Importance"]), reverse=True)
sp_df = pd.DataFrame(sp_rows[:15])
A(tbl(sp_df, float_fmt=".9f"))

A(h2("Figure 05  - GAT attention weight matrix"))
A("Entry [row, col]: mean attention weight that node 'row' (target)")
A("places on node 'col' (source) when computing its spatial embedding.")
A("Rows sum to 1 for each target node. Source: analysis/attention_weights.csv\n")
A(matrix_tbl(attn.round(4), float_fmt=".4f"))

A(h2("Figure 06  - Per-node detail (seasonal medians, observed only)"))
A("Temperature (°C)  - observed values only (imputed masked out)\n")
tmp_tbl = load_sensor_seasonal("temp_c")
if not tmp_tbl.empty:
    A(tbl(tmp_tbl, float_fmt=".2f"))
A("\nSalinity (PPT)\n")
sal_tbl = load_sensor_seasonal("sal_ppt")
if not sal_tbl.empty:
    A(tbl(sal_tbl, float_fmt=".2f"))
A("\nDissolved Oxygen mg/L\n")
do_tbl = load_sensor_seasonal("odo_mgL")
if not do_tbl.empty:
    A(tbl(do_tbl, float_fmt=".2f"))

A(h2("Figure 07  - Monthly salinity boxplot (median per calendar month)"))
A("Pooled observed salinity across all 7 nodes, by calendar month.\n")
sal_month_rows = []
for name, fname in IMPUTED_FILES.items():
    df = pd.read_csv(IMPUTED_DIR / fname, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    obs_col = "sal_ppt_observed"
    if obs_col in df.columns:
        df.loc[df[obs_col] == 0, "sal_ppt"] = np.nan
    df["month"] = df.index.month
    sal_month_rows.append(df[["month","sal_ppt"]])
sal_all = pd.concat(sal_month_rows)
sal_stats = sal_all.groupby("month")["sal_ppt"].agg(
    Median="median", Q25=lambda x: x.quantile(0.25),
    Q75=lambda x: x.quantile(0.75), Min="min", Max="max", N="count"
).round(2).reset_index()
sal_stats.columns = ["Month","Median","Q25","Q75","Min","Max","N"]
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
sal_stats["Month"] = sal_stats["Month"].map(MONTH_NAMES)
A(tbl(sal_stats, float_fmt=".2f"))

# ============================================================
A(h1("SECTION 2  - PHYSICAL FCM  (Figures 08–11)"))
# ============================================================
A("Daily resolution, 2025-2026. Forcing: Net Water (Rain-PET), temp_min,")
A("temp_max. Optimal lag = 1 day (selected by held-out validation MSE).")

A(h2("Figures 08 & 09  - Physical FCM weight matrix and causal graph"))
A("Source: analysis/fcm_weights_physical.csv")
A("Full 11×11 weight matrix (rows=source, cols=target):\n")
A(matrix_tbl(W_phys.round(3), float_fmt=".3f"))
A("\nTop causal edges (|weight| > 0.05):\n")
A(tbl(fcm_edge_table(W_phys, top_n=20), float_fmt=".3f"))

A(h2("Figure 10  - FCM scenario projections (summary)"))
A("Scenarios applied to Physical FCM; values are normalised activations [0,1].")
A("Shown: direction of change in each variable after 60-day simulation.\n")

# Rerun scenario computation for the table
try:
    import sys; sys.path.insert(0, ".")
    from fcm import (fcm_simulate, PHYS_CONCEPTS, N_FORCING, CONCEPT_LABELS,
                     load_physical_concept_timeseries)
    phys_daily = load_physical_concept_timeseries()
    C = len(PHYS_CONCEPTS)
    arr = phys_daily[PHYS_CONCEPTS].values.astype(np.float64)
    col_min   = np.nanmin(arr, axis=0)
    col_range = np.nanmax(arr, axis=0) - col_min
    col_range[col_range == 0] = 1.0
    arr_norm = (arr - col_min) / col_range
    W_p_np = W_phys.values.astype(np.float64)
    baseline = np.nanmean(arr_norm, axis=0)
    baseline[np.isnan(baseline)] = 0.5
    p5  = np.nanpercentile(arr_norm, 5,  axis=0)
    p95 = np.nanpercentile(arr_norm, 95, axis=0)
    scenarios = {
        "Baseline":   baseline.copy(),
        "Heavy Rain": baseline.copy(),
        "Heat Wave":  baseline.copy(),
        "Cold & Dry": baseline.copy(),
    }
    scenarios["Heavy Rain"][:N_FORCING] = [p95[0], baseline[1], baseline[2]]
    scenarios["Heat Wave"][:N_FORCING]  = [baseline[0], p95[1],  p95[2]]
    scenarios["Cold & Dry"][:N_FORCING] = [p5[0],  p5[1],  p5[2]]
    scen_rows = []
    labels = [CONCEPT_LABELS.get(c,c) for c in PHYS_CONCEPTS]
    for sname, A0 in scenarios.items():
        traj = fcm_simulate(W_p_np, A0.copy(), list(range(N_FORCING)),
                             A0[:N_FORCING], n_steps=60)
        final = traj[-1]
        for i, lab in enumerate(labels):
            if i < N_FORCING:
                continue
            scen_rows.append({"Concept": lab, "Scenario": sname,
                               "Final (norm)": round(float(final[i]),3),
                               "vs Baseline": round(float(final[i]-traj[0][i]),3)})
    scen_df = pd.DataFrame(scen_rows)
    scen_pivot = scen_df.pivot(index="Concept", columns="Scenario",
                                values="vs Baseline").reset_index()
    A(tbl(scen_pivot, float_fmt=".3f"))
except Exception as e:
    A(f"  (could not compute scenario table: {e})")

A(h2("Figure 11  - FCM influence ranking"))
A("Net influence = sum of outgoing |weights| - sum of incoming |weights|.")
A("Positive = net driver; Negative = net receiver.\n")
inf_rows = []
for c in W_phys.index:
    out_inf = W_phys.loc[c].abs().sum() - abs(W_phys.loc[c, c])
    in_inf  = W_phys[c].abs().sum()    - abs(W_phys.loc[c, c])
    inf_rows.append({"Concept": c, "Outgoing": round(float(out_inf),3),
                     "Incoming": round(float(in_inf),3),
                     "Net (Driver+)": round(float(out_inf - in_inf),3)})
inf_df = pd.DataFrame(inf_rows).sort_values("Net (Driver+)", ascending=False)
A(tbl(inf_df.reset_index(drop=True), float_fmt=".3f"))

# ============================================================
A(h1("SECTION 3  - NUTRIENT FCM  (Figures 12–15)"))
# ============================================================
A("Monthly resolution, 2021-2026. 16 concepts = 3 forcing + 8 sensor")
A("+ 5 nutrient extras (pH, Chl-a, Secchi, NO2+NO3, DIN).")
A("Optimal lag = 1 month (selected by held-out validation MSE).")

A(h2("Figures 12 & 13  - Nutrient FCM weight matrix and causal graph"))
A("Source: analysis/fcm_weights_nutrient.csv")
A("Full 16×16 weight matrix (rows=source, cols=target):\n")
A(matrix_tbl(W_nutr.round(3), float_fmt=".3f"))
A("\nTop causal edges (|weight| > 0.05):\n")
A(tbl(fcm_edge_table(W_nutr, top_n=25), float_fmt=".3f"))

A(h2("Figures 14–15  - Scenario projections and influence ranking (Nutrient FCM)"))
try:
    from fcm import (NUTR_CONCEPTS, N_NUTR, load_nutrient_concept_timeseries)
    nutr_monthly = load_nutrient_concept_timeseries()
    arr_n = nutr_monthly[NUTR_CONCEPTS].values.astype(np.float64)
    cmin_n   = np.nanmin(arr_n, axis=0)
    cr_n     = np.nanmax(arr_n, axis=0) - cmin_n
    cr_n[cr_n == 0] = 1.0
    arr_norm_n = (arr_n - cmin_n) / cr_n
    W_n_np   = W_nutr.values.astype(np.float64)
    base_n   = np.nanmean(arr_norm_n, axis=0)
    base_n[np.isnan(base_n)] = 0.5
    p5_n     = np.nanpercentile(arr_norm_n, 5, axis=0)
    p95_n    = np.nanpercentile(arr_norm_n, 95, axis=0)
    scens_n  = {
        "Baseline":   base_n.copy(),
        "Heavy Rain": base_n.copy(),
        "Heat Wave":  base_n.copy(),
        "Cold & Dry": base_n.copy(),
    }
    scens_n["Heavy Rain"][:N_FORCING] = [p95_n[0], base_n[1], base_n[2]]
    scens_n["Heat Wave"][:N_FORCING]  = [base_n[0], p95_n[1], p95_n[2]]
    scens_n["Cold & Dry"][:N_FORCING] = [p5_n[0], p5_n[1], p5_n[2]]
    rows_n = []
    labels_n = [CONCEPT_LABELS.get(c,c) for c in NUTR_CONCEPTS]
    for sname, A0 in scens_n.items():
        traj = fcm_simulate(W_n_np, A0.copy(), list(range(N_FORCING)),
                             A0[:N_FORCING], n_steps=60)
        for i, lab in enumerate(labels_n):
            if i < N_FORCING:
                continue
            rows_n.append({"Concept": lab, "Scenario": sname,
                            "vs Baseline": round(float(traj[-1][i]-traj[0][i]),3)})
    piv_n = pd.DataFrame(rows_n).pivot(index="Concept", columns="Scenario",
                                        values="vs Baseline").reset_index()
    A("\nScenario change vs baseline (normalised [0,1] space):\n")
    A(tbl(piv_n, float_fmt=".3f"))
except Exception as e:
    A(f"  (could not compute nutrient scenario table: {e})")

A("\nNutrient FCM influence ranking:\n")
inf_rows_n = []
for c in W_nutr.index:
    out_inf = W_nutr.loc[c].abs().sum() - abs(W_nutr.loc[c, c])
    in_inf  = W_nutr[c].abs().sum()     - abs(W_nutr.loc[c, c])
    inf_rows_n.append({"Concept": c, "Outgoing": round(float(out_inf),3),
                        "Incoming": round(float(in_inf),3),
                        "Net (Driver+)": round(float(out_inf - in_inf),3)})
inf_n_df = pd.DataFrame(inf_rows_n).sort_values("Net (Driver+)", ascending=False)
A(tbl(inf_n_df.reset_index(drop=True), float_fmt=".3f"))

# ============================================================
A(h1("SECTION 4  - DIE-OFF EVENT ANALYSIS  (Figures 16–20)"))
# ============================================================

A(h2("Figure 16  - Anomaly Z-scores at die-off events"))
A("Source: analysis/dieoff_anomalies.csv\n")
if not da.empty:
    A(tbl(da.round(3).reset_index(drop=True), float_fmt=".3f"))
else:
    A("  (dieoff_anomalies.csv is empty)")

A(h2("Figure 17  - Site values at event months (spatial map underlying data)"))
grab_raw = pd.read_csv(GRAB_CSV, encoding="latin-1")
grab_raw = grab_raw[grab_raw["sample_type"] == "Surface"].copy()
grab_raw = grab_raw[~grab_raw["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()
grab_raw["date"] = pd.to_datetime(grab_raw["date"])
grab_raw["site_name"] = grab_raw["site_name"].str.replace(r"^(GOC)(\d)", r"\1-\2", regex=True)
for event_month, event_label in [("2021-09", "Sept 2021"), ("2022-10", "Oct 2022")]:
    yr, mo = map(int, event_month.split("-"))
    evdf = grab_raw[(grab_raw["date"].dt.year==yr) & (grab_raw["date"].dt.month==mo)].copy()
    cols = ["site_name","site_type","lat_dec","lon_dec","do_per","do_mgL","nh4","din","sal"]
    evdf = evdf[[c for c in cols if c in evdf.columns]].drop_duplicates("site_name")
    evdf = evdf.sort_values("lat_dec", ascending=False).reset_index(drop=True)
    for c in ["do_per","do_mgL","nh4","din","sal"]:
        if c in evdf.columns:
            evdf[c] = pd.to_numeric(evdf[c], errors="coerce").round(2)
    A(f"\n  {event_label}  - all sampled sites:\n")
    A(tbl(evdf, float_fmt=".2f"))

A(h2("Figure 18  - Pre-event lead-up (3-month window, key sites)"))
A("Monthly grab-sample medians at canal and bay sites, 3 months before event.\n")
key_sites  = ["LR01","MR01","GOC-014","BB14","BB25"]
lead_vars  = ["do_per","nh4","din","sal","temp"]
for event_month, months_before, label in [
        ("2021-09", ["2021-07","2021-08","2021-09"], "Sept 2021"),
        ("2022-10", ["2022-08","2022-09","2022-10"], "Oct 2022"),
]:
    rows_l = []
    for ym in months_before:
        yr, mo = map(int, ym.split("-"))
        sub = grab_raw[(grab_raw["date"].dt.year==yr) & (grab_raw["date"].dt.month==mo) &
                       (grab_raw["site_name"].isin(key_sites))].copy()
        for _, r in sub.iterrows():
            row_d = {"Month": ym, "Site": r["site_name"]}
            for v in lead_vars:
                row_d[v] = round(float(pd.to_numeric(r.get(v, np.nan), errors="coerce")), 2) \
                            if not pd.isna(r.get(v, np.nan)) else np.nan
            rows_l.append(row_d)
    ld_df = pd.DataFrame(rows_l).sort_values(["Month","Site"]).reset_index(drop=True)
    A(f"\n  {label} lead-up:\n")
    A(tbl(ld_df, float_fmt=".2f"))

# ============================================================
A(h1("SECTION 5  - FCM VALIDATION  (Figures 21–24)"))
# ============================================================

A(h2("Figure 21  - One-step-ahead hindcast metrics"))
A("R² measured on genuinely observed data points only (not NaN-filled).")
A("R² > 0.5 = FCM has real predictive skill; < 0 = worse than mean prediction.")
A("Source: analysis/fcm_hindcast_metrics.csv\n")
hm_disp = hm.sort_values("R2", ascending=False).reset_index(drop=True)
hm_disp["R2"]   = hm_disp["R2"].round(3)
hm_disp["RMSE"] = hm_disp["RMSE"].round(3)
hm_disp["MAE"]  = hm_disp["MAE"].round(3)
A(tbl(hm_disp, float_fmt=".3f"))

A(h2("Figure 22  - Hindcast time-series (monthly observed vs predicted)"))
A("Predicted values computed as: A_pred(t+1) = sigmoid(W^T * A_obs(t))")
A("then de-normalised. Table shows first/last 12 months of salinity\n"
  "(the only concept with positive R²) as representative example.\n")
try:
    from fcm import (load_nutrient_concept_timeseries, learn_fcm_weights,
                     select_optimal_lag, NUTR_CONCEPTS, N_FORCING, _sigmoid)
    ts = load_nutrient_concept_timeseries()
    best_lag = select_optimal_lag(ts, NUTR_CONCEPTS, N_FORCING,
                                   [1,2,3], 1.5, tag="v2-report")
    W_v, arr_n2, cmin_v, cr_v = learn_fcm_weights(
        ts, NUTR_CONCEPTS, N_FORCING, 1.5, best_lag, "v2-report")
    ts_clean = ts[NUTR_CONCEPTS].dropna(
        subset=[NUTR_CONCEPTS[i] for i in range(N_FORCING, len(NUTR_CONCEPTS))], how="all")
    arr_c = ts_clean.values.astype(np.float64)
    arr_norm2 = (arr_c - cmin_v) / np.where(cr_v==0, 1, cr_v)
    sal_idx = NUTR_CONCEPTS.index("sal_ppt")
    pred_list = []
    for t in range(best_lag, len(arr_norm2)):
        A_t = arr_norm2[t - best_lag].copy()
        A_t[np.isnan(A_t)] = 0.5
        A_pred = _sigmoid(W_v.T @ A_t)
        obs    = arr_norm2[t, sal_idx]
        pred_list.append({
            "Month": str(ts_clean.index[t])[:7],
            "Observed (norm)": round(float(obs), 3) if not np.isnan(obs) else None,
            "Predicted (norm)": round(float(A_pred[sal_idx]), 3),
        })
    pred_df = pd.DataFrame(pred_list).dropna(subset=["Observed (norm)"])
    A(tbl(pred_df.reset_index(drop=True), float_fmt=".3f"))
except Exception as e:
    A(f"  (could not compute hindcast table: {e})")

A(h2("Figures 23-24  - Event backcasts (Sept 2021, Oct 2022)"))
A("Qualitative summary: FCM seeded at event-month observed conditions,")
A("run forward 6 months. See visualizations/23 and 24 for plotted results.")
A("The FCM predicts moderate salinity/temperature changes but cannot")
A("capture the acute DO crash because DO has negative hindcast R².")

# ============================================================
A(h1("SECTION 6  - GEOGRAPHIC MAPS: SENSOR STATIONS  (Figures 25–29)"))
# ============================================================

for feat, label, unit in [
        ("temp_c",        "Temperature", "°C"),
        ("sal_ppt",       "Salinity",    "PPT"),
        ("odo_mgL",       "DO",          "mg/L"),
        ("turbidity_fnu", "Turbidity",   "FNU"),
]:
    A(h2(f"Figure  - {label} ({unit}) seasonal medians (observed only)"))
    A(f"Rows = stations, columns = year+season combinations.")
    A(f"Grey in the chart = NaN here.\n")
    t = load_sensor_seasonal(feat)
    if not t.empty:
        A(tbl(t, float_fmt=".2f"))
    else:
        A("  (no observed data)")

# ============================================================
A(h1("SECTION 7  - EXTENDED MAPS 2021–2026  (Figures 30–36)"))
# ============================================================

A(h2("Figures 30–32  - Combined sensor + grab seasonal medians"))
A("Sensor stations (2025-2026) and grab-sample sites (2021-2024) combined.")
A("\nFigure 30  - Water Temperature at open-bay GRAB sites (°C)\n")
t30 = load_grab_seasonal("temp", site_types=["Biscayne Bay"])
if not t30.empty:
    A(tbl(t30, float_fmt=".1f"))

A("\nFigure 31  - Salinity at ALL grab sites (PPT)\n")
t31 = load_grab_seasonal("sal")
if not t31.empty:
    A(tbl(t31.iloc[:, :9], float_fmt=".1f"))   # first 8 cols to keep width manageable
    A("  (truncated to first 8 year+season columns; full data in CSV)")

A("\nFigure 32  - DO mg/L at ALL grab sites\n")
t32 = load_grab_seasonal("do_mgL")
if not t32.empty:
    A(tbl(t32.iloc[:, :9], float_fmt=".1f"))
    A("  (truncated to first 8 year+season columns)")

A(h2("Figures 33–35  - Nutrient seasonal medians (grab sites, 2021-2024)"))
for grab_col, label in [("ph","pH"), ("chl_exo_ugL","Chl-a (µg/L)"), ("din","DIN (µmol/L)")]:
    A(f"\nFigure  - {label}\n")
    nt = load_grab_seasonal(grab_col)
    if not nt.empty:
        A(tbl(nt.iloc[:, :9], float_fmt=".2f"))
        A("  (truncated to first 8 year+season columns)")

A(h2("Figure 36  - Canal stress: annual NH4 and DO% medians by site"))
A("Source: grab sample data, all surface samples.\n")
grab_raw["nh4"] = pd.to_numeric(grab_raw["nh4"], errors="coerce")
grab_raw["do_per"] = pd.to_numeric(grab_raw["do_per"], errors="coerce")
stress = grab_raw.groupby(["site_name","site_type",
                            grab_raw["date"].dt.year.rename("year")]
                          )[["nh4","do_per"]].median().round(2)
stress.columns = ["NH4 (µmol/L)","DO (%Sat)"]
stress = stress.reset_index().rename(columns={"year":"Year"})
stress = stress.sort_values(["Year","NH4 (µmol/L)"], ascending=[True,False])
A(tbl(stress.reset_index(drop=True), float_fmt=".2f"))

# ── write ────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "figures_summary_v2.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(blocks))

print(f"Report v2 written to {out_path}  ({len(blocks)} blocks,"
      f" {sum(b.count(chr(10)) for b in blocks) + len(blocks)} lines)")
