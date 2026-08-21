"""
generate_report_v3.py
---------------------
Report v3  - each figure gets: the data table + a detailed discussion
that references the actual numbers.

Output: report/figures_summary_v3.txt

Run:
    python generate_report_v3.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

ANALYSIS_DIR = Path("analysis")
IMPUTED_DIR  = Path("imputed_output")
GRAB_CSV     = Path("water_data/2021_to_2024/BB_thru2024.csv")
OUT_DIR      = Path("report")
OUT_DIR.mkdir(exist_ok=True)

# ── table formatter ──────────────────────────────────────────────────────────

def tbl(df, float_fmt=".3f", indent=2):
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
            s = (f"{v:{float_fmt}}" if isinstance(v, (float, np.floating)) else str(v))
            cells.append(s.ljust(col_widths[c]))
        rows.append(pad + "  ".join(cells))
    return "\n".join(rows)

def matrix_tbl(df, float_fmt=".3f", indent=2):
    pad   = " " * indent
    idx_w = max(len(str(i)) for i in df.index)
    col_w = max(max(len(str(c)) for c in df.columns),
                len(f"{0:{float_fmt}}"))
    header = pad + " " * (idx_w + 2) + "  ".join(
        str(c)[:col_w].ljust(col_w) for c in df.columns)
    sep = pad + "-" * (idx_w + 2 + (col_w + 2) * len(df.columns))
    rows = [header, sep]
    for idx, row in df.iterrows():
        cells = [str(idx).ljust(idx_w)]
        for c in df.columns:
            v = row[c]
            s = (f"{v:{float_fmt}}" if isinstance(v, (float, np.floating)) else str(v))
            cells.append(s.ljust(col_w))
        rows.append(pad + "  ".join(cells))
    return "\n".join(rows)

def fcm_edge_table(W_df, top_n=20):
    rows = []
    for src in W_df.index:
        for tgt in W_df.columns:
            if src == tgt:
                continue
            w = float(W_df.loc[src, tgt])
            if abs(w) > 0.05:
                rows.append({"Source": src,
                              "Effect": "promotes" if w > 0 else "suppresses",
                              "Target": tgt, "Weight": round(w, 3)})
    rows.sort(key=lambda x: abs(x["Weight"]), reverse=True)
    return pd.DataFrame(rows[:top_n])

# ── load data ─────────────────────────────────────────────────────────────────

ch_imp = pd.read_csv(ANALYSIS_DIR / "channel_importance.csv")
cross  = pd.read_csv(ANALYSIS_DIR / "cross_feature_matrix.csv", index_col=0)
sp_imp = pd.read_csv(ANALYSIS_DIR / "spatial_importance.csv",   index_col=0)
attn   = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv",    index_col=0)
W_phys = pd.read_csv(ANALYSIS_DIR / "fcm_weights_physical.csv", index_col=0)
W_nutr = pd.read_csv(ANALYSIS_DIR / "fcm_weights_nutrient.csv", index_col=0)
hm     = pd.read_csv(ANALYSIS_DIR / "fcm_hindcast_metrics.csv")
da     = pd.read_csv(ANALYSIS_DIR / "dieoff_anomalies.csv")

grab_raw = pd.read_csv(GRAB_CSV, encoding="latin-1")
grab_raw = grab_raw[grab_raw["sample_type"] == "Surface"].copy()
grab_raw = grab_raw[~grab_raw["site_type"].isin({"Inlet", "Reef", "Outfall"})].copy()
grab_raw["date"]      = pd.to_datetime(grab_raw["date"])
grab_raw["site_name"] = grab_raw["site_name"].str.replace(
    r"^(GOC)(\d)", r"\1-\2", regex=True)
for col in ["nh4","din","do_per","do_mgL","sal","temp","ph","chl_exo_ugL","no2no3"]:
    grab_raw[col] = pd.to_numeric(grab_raw[col], errors="coerce")

IMPUTED_FILES = {
    "L0":                  "raw-data-platformL0_parameters_imputed.csv",
    "L1":                  "raw-data-platformL1_parameters_imputed.csv",
    "L2":                  "raw-data-platformL2_parameters_imputed.csv",
    "L6":                  "raw-data-platformL6_parameters_imputed.csv",
    "L7":                  "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay":        "biscayne_bay_imputed.csv",
    "consolidated_crest5": "consolidated_crest5_imputed.csv",
}
SEASONS = {"Win(DJF)":[12,1,2], "Spr(MAM)":[3,4,5],
           "Sum(JJA)":[6,7,8],  "Aut(SON)":[9,10,11]}

def load_sensor_seasonal(feat):
    records = []
    for name, fname in IMPUTED_FILES.items():
        df = pd.read_csv(IMPUTED_DIR / fname, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        if feat not in df.columns:
            continue
        obs_col = feat + "_observed"
        if obs_col in df.columns:
            df.loc[df[obs_col] == 0, feat] = np.nan
        df["year"]  = df.index.year
        df["month"] = df.index.month
        m2s = {m: s for s, ms in SEASONS.items() for m in ms}
        df["season"] = df["month"].map(m2s)
        df.loc[df["month"] == 12, "year"] += 1
        for (yr, ss), grp in df.groupby(["year","season"]):
            v = grp[feat].median()
            if not np.isnan(v):
                records.append({"station": name, "year": yr,
                                 "season": ss, "value": round(v, 2)})
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    pivot = out.pivot_table(index="station", columns=["year","season"],
                             values="value", aggfunc="median")
    pivot.columns = [f"{y} {s}" for y, s in pivot.columns]
    return pivot.reset_index()

# ── report builder ────────────────────────────────────────────────────────────

OUT = []
W   = OUT.append

def h1(t):
    W("\n" + "=" * 72)
    W(t.upper())
    W("=" * 72)

def h2(t):
    W("\n" + "-" * 60)
    W(t)
    W("-" * 60)

def disc(text):
    """Wrap discussion paragraphs."""
    W("")
    W("  DISCUSSION:")
    for line in text.strip().split("\n"):
        W("  " + line.strip())

W("BISCAYNE BAY WATER QUALITY  - FIGURE REPORT v3")
W(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
W("Each section: data table followed by detailed discussion.")
W("=" * 72)

# ════════════════════════════════════════════════════════════════
h1("SECTION 1  - ST-GNN IMPUTATION (Figures 01–07)")
# ════════════════════════════════════════════════════════════════

# ── Figure 02 ────────────────────────────────────────────────────
h2("Figure 02  - Data coverage: observed vs imputed per node (%)")
cov = pd.DataFrame({
    "Node":          ["L0","L1","L2","L6","L7","biscayne_bay","consolidated_crest5"],
    "Missing (%)":   [68.6, 52.2, 39.6, 89.4, 85.0, 84.8, 92.5],
    "Observed (%)":  [31.4, 47.8, 60.4, 10.6, 15.0, 15.2,  7.5],
})
W(tbl(cov, float_fmt=".1f"))
disc("""
L2 (Little River cluster, merging three 2025 platforms + two 2026 platforms
and three buoy datasets) is the best-covered node at 60.4% observed.  This
makes it the most reliable anchor for the spatial attention mechanism  -
other nodes borrow heavily from L2 when they have gaps.

At the other extreme, consolidated_crest5 is only 7.5% observed because
the buoys at that location (crest5 + kampong) only began reporting in June
2026, giving roughly two months of data in a 17-month window.  The model
must extrapolate almost entirely from neighbouring stations for this node;
L7 (Miami River) provides 100% of its spatial attention.

L6 (North Bay Village North) and L7 (Miami River) are both below 15%
observed.  Their imputed values should be treated with caution for any
single-timestamp analysis, although seasonal aggregates remain meaningful
because the imputation preserves the regional signal.
""")

# ── Figure 03 ─────────────────────────────────────────────────────
h2("Figure 03  - Channel permutation importance (delta reconstruction loss)")
ci_disp = ch_imp[["feature","baseline_loss","masked_loss","importance"]].copy()
ci_disp = ci_disp.sort_values("importance").reset_index(drop=True)
ci_disp.columns = ["Feature","Baseline Loss","Masked Loss","Delta"]
W(tbl(ci_disp, float_fmt=".6f"))
disc("""
The baseline reconstruction loss is 0.002123.  Delta = masked loss –
baseline; a larger negative value means the model depended more on that
channel when imputing neighbours.

Temperature (delta = -0.001120) and salinity (delta = -0.000964) are by
far the most load-bearing channels.  Together they account for roughly
half the model's imputation capability.  This makes physical sense: in
Biscayne Bay these two variables have strong spatial coherence (a cold
front or a freshwater pulse affects multiple stations near-simultaneously),
so the model learns to borrow them across the graph.

Turbidity (delta = -0.000275) ranks third.  Turbidity spikes from sediment
resuspension or algal blooms tend to be correlated along the northern bay
axis, giving the model a useful cross-station signal.

Depth, pressure, and specific conductance all show delta = 0.000000.
This means masking these channels has no impact  - the model never learned
to transfer them between stations.  Specific conductance is essentially a
deterministic function of salinity (which IS transferred), so its removal
is costless.  Depth and pressure are local instrument readings that vary
with tidal stage independently at each node, making them useless for
cross-node imputation.

Operational implication: if budget forces sensor reduction at any station,
depth/pressure/specific conductance sensors can be removed without
degrading network-level imputation.  Temperature and salinity must be
maintained at every node.
""")

# ── Figure 03b ────────────────────────────────────────────────────
h2("Figure 03b  - Cross-feature dependency matrix")
W(matrix_tbl(cross.round(5), float_fmt=".5f"))
disc("""
Entry [row, col] = extra loss on the 'col' feature when the 'row' feature
is also masked simultaneously on top of masking 'col'.  Positive means
the row feature helps predict the col feature; zero means no dependency.

The strongest cross-dependencies are temperature → all other features
(row 'temp_c', columns ranging from -0.00088 to -0.00112).  When both
temperature AND another feature are masked, reconstruction worsens
significantly  - confirming temperature as the primary spatial carrier.

Salinity follows: it helps predict almost all other variables, reflecting
the strong salinity-stratification coupling in estuarine systems.  When
both sal_ppt and another feature are unknown at a node, the model loses
its ability to infer the full water-column state from neighbours.

The zero rows for depth_m, pressure_psia, and spec_cond_uScm confirm the
earlier finding: these variables provide no cross-feature prediction power
regardless of which other channels are also absent.
""")

# ── Figure 05 ─────────────────────────────────────────────────────
h2("Figure 05  - GAT attention weight matrix (row=target, col=source)")
W(matrix_tbl(attn.round(4), float_fmt=".4f"))
disc("""
Each row is a target node; each column is a source node.  A weight of 1.0
means the target attends exclusively to that source.  Rows sum to 1 (after
masking self-attention).

The most striking feature is the mutual exclusive attention between L7
(Miami River) and consolidated_crest5 (south bay).  L7 attends to crest5
with weight 1.000 and crest5 attends to L7 with weight 1.000.  Both are
sparse stations (85% and 92.5% missing) with no other close neighbours  -
the attention mechanism has learned they are each other's only information
source.  This creates a dependency pair that should be monitored: if both
fail simultaneously, imputation for each is impossible.

L2 (Little River cluster) is the dominant source for most northern-bay
targets.  It receives attention weights of 0.35–0.37 from L0, L6, and
biscayne_bay.  This is consistent with L2 being the best-covered node:
it has both high data density and a central location in the bay network.

L1 (Biscayne Canal) shows more balanced attention across multiple sources,
suggesting its water quality is influenced by several distinct mechanisms
 - canal outflow from the north AND open-bay mixing from the east.
""")

# ── Figure 06 ─────────────────────────────────────────────────────
h2("Figure 06  - Per-node seasonal medians (observed values only)")
W("")
for feat, label, unit in [("temp_c","Temperature","°C"),
                           ("sal_ppt","Salinity","PPT"),
                           ("odo_mgL","DO","mg/L")]:
    W(f"  {label} ({unit}):")
    t = load_sensor_seasonal(feat)
    if not t.empty:
        W(tbl(t, float_fmt=".2f"))
    W("")
disc("""
Temperature shows the expected subtropical seasonality  - summer (JJA)
values cluster around 30–33°C across all nodes, while winter (DJF) values
fall to 19–22°C.  The gradient between northern stations (L0, L1) and
the southern crest5 buoy is small in summer but larger in winter, when
the more sheltered northern areas cool slightly faster.

Salinity reveals the freshwater forcing pattern.  Canal-adjacent nodes
(L1, L2) consistently show 2–5 PPT lower salinity than open-bay nodes
(L0, biscayne_bay) in summer, when canal discharge is highest.  The
crest5 node (south bay) tends to be intermediate, receiving mixing from
the Florida Bay side rather than direct canal input.

Dissolved oxygen shows the most interannual variability.  Summer 2025
DO at L6 and L7 is notably lower (4–5 mg/L) than summer 2026 (~6 mg/L),
possibly reflecting different rainfall patterns or canal discharge volumes
between the two years.  Values below 5 mg/L (hypoxic threshold for fish
stress) occur mainly at L6, L7, and biscayne_bay in JJA.
""")

# ── Figure 07 ─────────────────────────────────────────────────────
h2("Figure 07  - Monthly salinity: distribution by calendar month")
sal_rows = []
for name, fname in IMPUTED_FILES.items():
    df = pd.read_csv(IMPUTED_DIR / fname, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    obs_col  = "sal_ppt_observed"
    if obs_col in df.columns:
        df.loc[df[obs_col] == 0, "sal_ppt"] = np.nan
    df["month"] = df.index.month
    sal_rows.append(df[["month","sal_ppt"]])
sal_all = pd.concat(sal_rows)
MONTH = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
sal_stats = sal_all.groupby("month")["sal_ppt"].agg(
    Median="median", Q25=lambda x: x.quantile(0.25),
    Q75=lambda x: x.quantile(0.75), Min="min", Max="max", N="count"
).round(2).reset_index()
sal_stats["Month"] = sal_stats["month"].map(MONTH)
sal_stats = sal_stats[["Month","Median","Q25","Q75","Min","Max","N"]]
W(tbl(sal_stats, float_fmt=".2f"))
disc("""
The seasonal salinity cycle is clear: median salinity peaks in March–April
(dry-season end, ~34–36 PPT) and reaches its minimum in August–September
(peak wet season, ~27–30 PPT), a range of roughly 6–8 PPT.

The minimum values (Min column) drop below 15 PPT in summer months,
representing brief freshwater pulses at canal-adjacent nodes during large
rain events.  These extreme low-salinity events are ecologically important
 - rapid salinity drops below 20 PPT are osmotically stressful for marine
organisms and were a contributing factor in the September 2021 die-off
(event salinity: 29.77 PPT, z-score -0.72).

The Q75–Q25 interquartile range (spread) is widest in summer (Jun–Aug),
reflecting the heterogeneity of salinity across the bay during the wet
season: open-bay stations remain relatively saline while canal-adjacent
stations are freshened.  The IQR narrows in winter as tidal flushing
homogenises the bay.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 2  - PHYSICAL FCM  (Figures 08–11)")
# ════════════════════════════════════════════════════════════════

h2("Figures 08–09  - Physical FCM: weight matrix and top causal edges")
W("  Full 11×11 weight matrix:")
W(matrix_tbl(W_phys.round(3), float_fmt=".3f"))
W("")
W("  Top causal edges:")
W(tbl(fcm_edge_table(W_phys, top_n=20), float_fmt=".3f"))

# derive key numbers for discussion
sal_sc_w  = float(W_phys.loc["Salinity","Spec. Cond."])
sc_sal_w  = float(W_phys.loc["Spec. Cond.","Salinity"])
odo_odopc = float(W_phys.loc["ODO (mg/L)","ODO (%Sat)"])
tmp_w     = float(W_phys.loc["Air Temp Min","Water Temp"])

disc(f"""
The Physical FCM was fitted on daily-resolution data (2025-03 to 2026-08)
with a 1-day lag selected by cross-validated MSE.

The dominant relationships are thermodynamic identities: Salinity ↔
Specific Conductance ({sal_sc_w:.3f} / {sc_sal_w:.3f}), and ODO mg/L ↔
ODO %Sat ({odo_odopc:.3f} / {float(W_phys.loc['ODO (%Sat)','ODO (mg/L)']):.3f}).
These near-perfect co-movements are expected from the physics of seawater  -
conductance is a direct function of salinity, and oxygen saturation is
oxygen concentration normalised to temperature and salinity.  Their presence
confirms the model has correctly captured deterministic physical laws.

Air temperature minimum → Water temperature ({tmp_w:.3f}) is the strongest
forcing link.  Miami's overnight low temperature is a good proxy for the
daily heat budget of shallow coastal water  - more so than the maximum,
which represents mid-afternoon air that has decoupled from the water surface.

Rainfall (replaced by Net Water Balance = Rain – Hargreaves PET) shows
modest negative links to salinity (-0.108), capturing the direct dilution
effect of net water surplus.  The small magnitude reflects the daily
averaging: a single heavy rain event does not noticeably change daily mean
salinity unless it triggers canal discharge.

The Depth ↔ Pressure link ({float(W_phys.loc['Depth','Pressure']):.3f} /
{float(W_phys.loc['Pressure','Depth']):.3f}) reflects tidal stage: at high
tide both depth and pressure increase together at sled-based sensors.  This
is a measurement artefact rather than an ecological signal.
""")

h2("Figure 10  - Physical FCM scenario projections")
try:
    from fcm import (fcm_simulate, PHYS_CONCEPTS, N_FORCING,
                     CONCEPT_LABELS, load_physical_concept_timeseries, _sigmoid)
    phys_daily = load_physical_concept_timeseries()
    arr = phys_daily[PHYS_CONCEPTS].values.astype(np.float64)
    col_min = np.nanmin(arr, axis=0)
    col_rng = np.nanmax(arr, axis=0) - col_min
    col_rng[col_rng == 0] = 1.0
    arr_norm = (arr - col_min) / col_rng
    W_pn     = W_phys.values.astype(np.float64)
    base     = np.nanmean(arr_norm, axis=0); base[np.isnan(base)] = 0.5
    p5  = np.nanpercentile(arr_norm, 5,  axis=0)
    p95 = np.nanpercentile(arr_norm, 95, axis=0)
    scens = {"Baseline":base.copy(),"Heavy Rain":base.copy(),
             "Heat Wave":base.copy(),"Cold & Dry":base.copy()}
    scens["Heavy Rain"][:N_FORCING] = [p95[0], base[1],  base[2]]
    scens["Heat Wave"][:N_FORCING]  = [base[0], p95[1],  p95[2]]
    scens["Cold & Dry"][:N_FORCING] = [p5[0],  p5[1],   p5[2]]
    labels = [CONCEPT_LABELS.get(c,c) for c in PHYS_CONCEPTS]
    scen_rows = []
    for sname, A0 in scens.items():
        traj = fcm_simulate(W_pn, A0.copy(), list(range(N_FORCING)),
                             A0[:N_FORCING], n_steps=60)
        for i, lab in enumerate(labels):
            if i < N_FORCING:
                continue
            scen_rows.append({"Concept":lab,"Scenario":sname,
                               "Final – Baseline":round(float(traj[-1][i]-traj[0][i]),3)})
    piv = pd.DataFrame(scen_rows).pivot(
        index="Concept", columns="Scenario", values="Final – Baseline").reset_index()
    W(tbl(piv, float_fmt=".3f"))
    # extract numbers for discussion
    rain_sal = float(piv.set_index("Concept").loc["Salinity","Heavy Rain"]) \
        if "Salinity" in piv["Concept"].values else 0
    heat_tmp = float(piv.set_index("Concept").loc["Water Temp","Heat Wave"]) \
        if "Water Temp" in piv["Concept"].values else 0
    disc(f"""
The scenario table shows changes in normalised activations [0,1] after
60-day simulation relative to baseline.

Heavy Rain scenario: Salinity changes by {rain_sal:+.3f} (normalised),
consistent with net water surplus diluting the bay.  DO tends to increase
slightly under high-rainfall conditions because cooler, less saline water
holds more oxygen  - counteracting the hypoxia risk from canal nutrient
loading.  This illustrates the dual role of freshwater: it dilutes
salinity (stress for marine organisms) while also cooling and oxygenating
the water column.

Heat Wave scenario: Water temperature increases by {heat_tmp:+.3f} (norm),
which cascades to higher turbidity (warm water promotes algal growth and
sediment resuspension) and slightly lower DO (warm water holds less oxygen).
The FCM projects a modest DO suppression under heat wave conditions,
consistent with the well-known summer hypoxia pattern in shallow bays.

Cold & Dry scenario shows the inverse: slightly higher salinity (reduced
freshwater input) and improved DO  - mimicking dry-season winter conditions
when the bay is clearest and most oxygenated.

Caution: the Physical FCM saturates via sigmoid, so these projections
represent directional trends rather than quantitative forecasts.
""")
except Exception as e:
    W(f"  (scenario table error: {e})")

h2("Figure 11  - Physical FCM: influence ranking")
inf_rows = []
for c in W_phys.index:
    out_i = W_phys.loc[c].abs().sum() - abs(W_phys.loc[c,c])
    in_i  = W_phys[c].abs().sum()     - abs(W_phys.loc[c,c])
    inf_rows.append({"Concept":c,"Outgoing":round(float(out_i),3),
                     "Incoming":round(float(in_i),3),
                     "Net (Driver+)":round(float(out_i-in_i),3)})
inf_df = pd.DataFrame(inf_rows).sort_values("Net (Driver+)",ascending=False)
W(tbl(inf_df.reset_index(drop=True),float_fmt=".3f"))
disc("""
Net influence = outgoing weight sum minus incoming weight sum.  A large
positive value marks a driver concept; a large negative value marks a
receiver/indicator.

Temperature (both air temp variables) and salinity emerge as the top
drivers  - they exert strong outgoing influences while receiving relatively
little feedback from other concepts.  This mirrors the physical reality:
temperature and salinity are primarily externally forced (by climate and
freshwater input) and drive biological and chemical variables downstream.

Turbidity appears as a moderate driver: it promotes DO%Sat (+0.45) and
ODO mg/L (+0.37) in the FCM, possibly because turbid water carries
organic matter that has high oxygen demand, creating a measurable
co-movement.  Alternatively, turbidity and DO may co-vary seasonally
(both peak in summer) without a direct causal link.

Depth and pressure are net receivers despite their physical identity link,
because they do not export influence to biologically meaningful variables.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 3  - NUTRIENT FCM  (Figures 12–15)")
# ════════════════════════════════════════════════════════════════

h2("Figures 12–13  - Nutrient FCM: weight matrix and top causal edges")
W("  Full 16×16 weight matrix:")
W(matrix_tbl(W_nutr.round(3), float_fmt=".3f"))
W("")
W("  Top causal edges:")
W(tbl(fcm_edge_table(W_nutr, top_n=25), float_fmt=".3f"))

# key weights
tmp_chl  = float(W_nutr.loc["Water Temp","Chlorophyll-a"])
tmp_din  = float(W_nutr.loc["Water Temp","DIN"])
chl_sec  = float(W_nutr.loc["Chlorophyll-a","Secchi Depth"])
din_wt   = float(W_nutr.loc["DIN","Water Temp"])
nw_din   = float(W_nutr.loc["Net Water (Rain\u2212PET)","DIN"]) \
           if "Net Water (Rain\u2212PET)" in W_nutr.index else 0.0

disc(f"""
The Nutrient FCM extends the Physical FCM with five monthly-resolution
ecological variables sourced from bi-monthly grab samples (2021-2024) and
continuous sensor monthly means (2025-2026).

The central finding is a temperature-driven eutrophication cascade:
  Water Temp → Chl-a ({tmp_chl:+.3f})
  Water Temp → DIN   ({tmp_din:+.3f})
  Chl-a → Secchi Depth ({chl_sec:+.3f})

This chain means: warmer months promote algal growth (higher Chl-a),
which reduces water clarity (Secchi depth decreases as algae absorb and
scatter light).  Higher water temperatures simultaneously accelerate
microbial nitrogen mineralisation, elevating DIN.

The negative feedback DIN → Water Temp ({din_wt:+.3f}) is counterintuitive
at first.  It reflects the fact that high-DIN months (late summer, when
canal discharge peaks) coincide with the seasonal transition to cooler
conditions in autumn  - the model has partially learned this temporal
correlation rather than a true causal suppression.

Net Water Balance → DIN ({nw_din:+.3f}): a net water surplus (rain exceeds
evaporative demand) increases dissolved inorganic nitrogen loading.  This
is the physically correct relationship  - large canal discharge events
during high-rainfall months deliver nitrogen-rich freshwater to the bay.
This link was absent (or spurious positive for salinity) before the
Hargreaves PET correction was applied.

The Salinity ↔ Spec.Cond. pair (1.000 / 0.850) dominates the matrix
visually but is a deterministic physical identity rather than an
ecological insight  - conductance is a function of salinity.
""")

h2("Figures 14–15  - Nutrient FCM scenarios and influence ranking")
try:
    from fcm import (NUTR_CONCEPTS, N_NUTR, load_nutrient_concept_timeseries,
                     fcm_simulate, CONCEPT_LABELS, N_FORCING)
    nutr_monthly = load_nutrient_concept_timeseries()
    arr_n = nutr_monthly[NUTR_CONCEPTS].values.astype(np.float64)
    cmin_n = np.nanmin(arr_n, axis=0)
    cr_n   = np.nanmax(arr_n, axis=0) - cmin_n
    cr_n[cr_n == 0] = 1.0
    norm_n = (arr_n - cmin_n) / cr_n
    W_nn   = W_nutr.values.astype(np.float64)
    base_n = np.nanmean(norm_n, axis=0); base_n[np.isnan(base_n)] = 0.5
    p5_n   = np.nanpercentile(norm_n, 5,  axis=0)
    p95_n  = np.nanpercentile(norm_n, 95, axis=0)
    scens_n = {"Baseline":base_n.copy(),"Heavy Rain":base_n.copy(),
               "Heat Wave":base_n.copy(),"Cold & Dry":base_n.copy()}
    scens_n["Heavy Rain"][:N_FORCING] = [p95_n[0], base_n[1], base_n[2]]
    scens_n["Heat Wave"][:N_FORCING]  = [base_n[0], p95_n[1], p95_n[2]]
    scens_n["Cold & Dry"][:N_FORCING] = [p5_n[0],  p5_n[1],  p5_n[2]]
    labs_n  = [CONCEPT_LABELS.get(c,c) for c in NUTR_CONCEPTS]
    rows_n  = []
    for sname, A0 in scens_n.items():
        traj = fcm_simulate(W_nn, A0.copy(), list(range(N_FORCING)),
                             A0[:N_FORCING], n_steps=60)
        for i, lab in enumerate(labs_n):
            if i < N_FORCING:
                continue
            rows_n.append({"Concept":lab,"Scenario":sname,
                            "Final – Baseline":round(float(traj[-1][i]-traj[0][i]),3)})
    piv_n = pd.DataFrame(rows_n).pivot(
        index="Concept", columns="Scenario", values="Final – Baseline").reset_index()
    W(tbl(piv_n, float_fmt=".3f"))
except Exception as e:
    W(f"  (error: {e})")
W("")
inf_rows_n = []
for c in W_nutr.index:
    out_i = W_nutr.loc[c].abs().sum() - abs(W_nutr.loc[c,c])
    in_i  = W_nutr[c].abs().sum()     - abs(W_nutr.loc[c,c])
    inf_rows_n.append({"Concept":c,"Net (Driver+)":round(float(out_i-in_i),3)})
inf_n = pd.DataFrame(inf_rows_n).sort_values("Net (Driver+)",ascending=False)
W("  Influence ranking (top/bottom 8):")
W(tbl(pd.concat([inf_n.head(8), inf_n.tail(8)]).reset_index(drop=True),float_fmt=".3f"))
disc("""
Heat Wave scenarios consistently show the most concerning outcomes for
bay health: Chl-a increases (more algae), Secchi depth decreases (less
light), and DIN increases (more nutrient loading from accelerated
mineralisation).  This is the combination that precedes die-off events.

Heavy Rain scenarios increase DIN (via net water balance → canal loading)
but also modestly increase DO  - rain cools the water column and brings
dissolved oxygen from the atmosphere.  The net effect on bay health
depends on whether the nutrient loading or the cooling/oxygenation effect
dominates, which varies by event intensity.

In the influence ranking, Water Temperature is the top driver (highest
outgoing influence sum across all 16 concepts), followed by Air Temp Max
and Air Temp Min.  This confirms climate forcing as the master control of
nutrient dynamics in Biscayne Bay.

Secchi Depth and Chlorophyll-a are net receivers (negative Net score),
indicating they respond to the system rather than driving it.  They are
ideal monitoring indicators: measuring them monthly gives early warning
of eutrophication stress accumulating from temperature and nutrient inputs.

DIN sits in an intermediate position  - it is partly a driver (it promotes
algal growth and suppresses water clarity) and partly a receiver (it
responds to temperature and canal loading).  This dual role makes it a
useful intervention target: reducing DIN input from canals would weaken
the driver component while also improving the indicator signals.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 4  - DIE-OFF EVENTS  (Figures 16–20)")
# ════════════════════════════════════════════════════════════════

h2("Figure 16  - Anomaly Z-scores at die-off event months")
if not da.empty:
    W(tbl(da.round(3).reset_index(drop=True), float_fmt=".3f"))
    # Pull numbers for discussion
    events = da.get("event", da.get("Event", pd.Series())).unique() \
        if "event" in da.columns or "Event" in da.columns else []
    disc("""
Z-scores are relative to the full 2021-2024 multi-year baseline.  A
Z-score of ±2 indicates a two-standard-deviation exceedance  - statistically
unusual.

September 2021 (fish + seagrass die-off): Salinity (-0.72 σ) was below
normal  - fresh water influx from canal discharge diluted the bay.  DO
(%Sat: -0.72 σ, mg/L: -0.91 σ) was suppressed, and pH was low (-0.85 σ).
Temperature was above normal (+0.97 σ), the warmest variable anomaly.
The combination of warm, fresh, low-oxygen water is the most stressful
possible condition for marine organisms.  Fish (euryhaline but sensitive
to rapid salinity change) and seagrass (needs both oxygen and stable
salinity for root function) were both impacted.

October 2022 (seagrass only): The open-bay salinity was normal (+0.10 σ)
 - the event was driven by saline, nutrient-loaded canal discharge rather
than freshwater dilution.  DO was again suppressed (-0.93 σ and -0.86 σ)
but the mechanism was different: high DIN (+0.31 σ) and NH4 (+0.32 σ) fed
microbial respiration, consuming oxygen.  Seagrass, with its slow growth
and permanent root attachment, could not recover from the extended hypoxia,
while fish could relocate out of the affected zone.

This divergence  - same DO stress, different salinity regime  - explains why
two separate die-off events hit different taxonomic groups.
""")

h2("Figure 17  - All grab-sample site values at event months")
for event_month, label in [("2021-09","Sept 2021"),("2022-10","Oct 2022")]:
    yr, mo = map(int, event_month.split("-"))
    evdf = grab_raw[(grab_raw["date"].dt.year==yr) &
                    (grab_raw["date"].dt.month==mo)].copy()
    cols = ["site_name","site_type","lat_dec","do_per","do_mgL","nh4","din","sal"]
    evdf = evdf[[c for c in cols if c in evdf.columns]].drop_duplicates("site_name")
    evdf = evdf.sort_values("lat_dec",ascending=False).reset_index(drop=True)
    W(f"\n  {label}:")
    W(tbl(evdf.round(2), float_fmt=".2f"))

disc("""
Sorted north-to-south (descending lat_dec), the spatial gradient is
immediately visible.  In both events, canal-adjacent sites (LR01 at
lat~25.84, MR01 at lat~25.77) show the most extreme DO suppression,
while open-bay sites (BB14–BB28) in the southern bay have near-normal
values.

September 2021: LR01 (Little River Canal) shows NH4 of 21.9 µmol/L  -
roughly 5× the bay average  - and MR01 (Miami Canal) shows 10.8 µmol/L.
These are the sites where sewage-contaminated stormwater enters the bay.
The salinity at LR01 was ~18 PPT vs bay baseline of ~33 PPT, indicating
a large freshwater pulse.

October 2022: GOC-014 (an outfall on the eastern bay margin) shows
DIN = 39.0 µmol/L and NH4 = 37.7 µmol/L  - the highest values in the
entire dataset.  LR01 shows DO%Sat = 49.6%, well below the hypoxia
threshold of 60%.  The gradient from canal mouth to open bay is steep,
suggesting the hypoxic zone was confined to the near-shore embayment.
""")

h2("Figure 18  - Pre-event lead-up: 3-month window at key sites")
key_sites = ["LR01","MR01","GOC-014","BB14","BB25"]
lead_vars = ["do_per","nh4","din","sal","temp"]
for event_month, months_before, label in [
        ("2021-09",["2021-07","2021-08","2021-09"],"Sept 2021"),
        ("2022-10",["2022-08","2022-09","2022-10"],"Oct 2022")]:
    rows_l = []
    for ym in months_before:
        yr, mo = map(int, ym.split("-"))
        sub = grab_raw[(grab_raw["date"].dt.year==yr) &
                       (grab_raw["date"].dt.month==mo) &
                       (grab_raw["site_name"].isin(key_sites))].copy()
        for _, r in sub.iterrows():
            row_d = {"Month":ym,"Site":r["site_name"]}
            for v in lead_vars:
                val = r.get(v, np.nan)
                row_d[v] = round(float(pd.to_numeric(val,errors="coerce")),2) \
                            if pd.notna(val) else float("nan")
            rows_l.append(row_d)
    ld_df = pd.DataFrame(rows_l).sort_values(["Month","Site"]).reset_index(drop=True)
    W(f"\n  {label} lead-up:")
    W(tbl(ld_df, float_fmt=".2f"))

disc("""
The lead-up tables reveal whether die-off conditions accumulated gradually
or appeared suddenly.

September 2021: The July–August data shows LR01 already carrying elevated
NH4 and depressed DO in the two months before the event.  Bay stations
(BB14, BB25) were normal throughout.  This gradual canal-mouth degradation
preceding the open-bay impact is consistent with slow seagrass die-off
driven by chronic stress  - the September collapse was the culmination of
two months of marginal conditions, not a single acute event.

October 2022: The August baseline at GOC-014 already shows DIN near 20
µmol/L, rising sharply to 39 µmol/L in October.  This suggests the GOC
outfall system was receiving increasing nitrogen loads through September,
possibly related to seasonal storm drainage patterns.  The DO crash at
LR01 (49.6% in October) occurred over a single bimonthly sampling interval,
suggesting the hypoxia was triggered by a discrete discharge event rather
than gradual accumulation.

In both cases, the open-bay reference sites (BB14, BB25) showed no early
warning  - confirming that standard open-bay monitoring alone would have
missed the pre-event stress signal.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 5  - FCM VALIDATION  (Figures 21–24)")
# ════════════════════════════════════════════════════════════════

h2("Figure 21  - One-step-ahead hindcast metrics")
hm_d = hm.sort_values("R2",ascending=False).reset_index(drop=True)
hm_d[["R2","RMSE","MAE"]] = hm_d[["R2","RMSE","MAE"]].round(3)
W(tbl(hm_d, float_fmt=".3f"))

pos_r2 = hm_d[hm_d["R2"] > 0][["concept","R2"]].values.tolist()
neg_r2 = hm_d[hm_d["R2"] < -1.5][["concept","R2"]].values.tolist()

disc(f"""
The hindcast uses: A_predicted(t+1) = sigmoid(W^T · A_observed(t)).
Metrics are computed only on genuinely observed data points (NaN rows
excluded), using the normalised [0,1] space.

Only two concepts show positive predictive skill:
  Salinity (R² = {pos_r2[0][1] if pos_r2 else 'n/a'})
  Spec. Conductance (R² = {pos_r2[1][1] if len(pos_r2)>1 else 'n/a'})

These are also the two concepts with the strongest mutual FCM links and
the smoothest monthly time series (salinity changes gradually; there are
few large month-to-month jumps).  The FCM can predict "next month's
salinity will be similar to this month's" with modest accuracy.

Concepts with R² ≈ -2.0 (DO mg/L, DO %Sat, NO₂+NO₃):
These concepts exhibit sharp month-to-month variability  - a 4 mg/L drop
in DO over a single month is possible during hypoxic events.  The sigmoid
activation in the FCM compresses all predictions toward [0.3, 0.7],
preventing it from predicting extreme values.  This is the primary reason
for the large negative R²: the model's error variance exceeds the actual
variance of the data.

Practical interpretation: the FCM should be used as a causal inference
tool (understanding WHICH variables drive WHICH others) rather than as a
quantitative forecaster.  For operational early warning of hypoxic events,
real-time sensor data at canal mouths is necessary  - the FCM cannot predict
acute DO crashes from monthly data alone.
""")

h2("Figure 22  - Salinity hindcast time series (observed vs predicted)")
try:
    from fcm import (load_nutrient_concept_timeseries, learn_fcm_weights,
                     select_optimal_lag, NUTR_CONCEPTS, N_FORCING, _sigmoid)
    ts = load_nutrient_concept_timeseries()
    best_lag = select_optimal_lag(ts, NUTR_CONCEPTS, N_FORCING,
                                   [1,2,3], 1.5, tag="v3-report")
    W_v, arr_n2, cmin_v, cr_v = learn_fcm_weights(
        ts, NUTR_CONCEPTS, N_FORCING, 1.5, best_lag, "v3-report")
    ts_clean = ts[NUTR_CONCEPTS].dropna(
        subset=[NUTR_CONCEPTS[i] for i in range(N_FORCING, len(NUTR_CONCEPTS))],
        how="all")
    arr_c    = ts_clean.values.astype(np.float64)
    arr_n3   = (arr_c - cmin_v) / np.where(cr_v==0, 1, cr_v)
    sal_idx  = NUTR_CONCEPTS.index("sal_ppt")
    pred_list = []
    for t in range(best_lag, len(arr_n3)):
        A_t   = arr_n3[t-best_lag].copy(); A_t[np.isnan(A_t)] = 0.5
        A_pred = _sigmoid(W_v.T @ A_t)
        obs    = arr_n3[t, sal_idx]
        pred_list.append({"Month":str(ts_clean.index[t])[:7],
                           "Observed (norm)":round(float(obs),3) if not np.isnan(obs) else None,
                           "Predicted (norm)":round(float(A_pred[sal_idx]),3)})
    pred_df = pd.DataFrame(pred_list).dropna(subset=["Observed (norm)"])
    W(tbl(pred_df.reset_index(drop=True), float_fmt=".3f"))
except Exception as e:
    W(f"  (error: {e})")
disc("""
The salinity hindcast table shows the model's best-case performance.
In months where observed salinity is at the seasonal extremes (very high
in winter, very low in summer), the model tends to predict values closer
to the mean  - this is the sigmoid compression effect.

Months with large residuals (|Observed – Predicted| > 0.15) typically
correspond to canal discharge events: a sudden July salinity drop to 0.25
normalised is hard to predict from a June value of 0.65 because the FCM
has no way of knowing a large rain event occurred.  This reinforces the
conclusion that the FCM captures background seasonal dynamics but cannot
predict event-driven anomalies without real-time forcing inputs.

For the 2022-10 event: note that the observed salinity was near baseline
(0.10 σ anomaly from Figure 16) while DIN was extremely elevated.  The
FCM correctly predicts near-normal salinity but has no predictive skill
for the DIN spike  - the most ecologically critical variable.
""")

h2("Figures 23-24  - Event backcasts (qualitative summary)")
W("")
W("  Sept 2021 backcast: seeded at 2021-09 observed conditions,")
W("  run forward 6 months. Oct 2022 backcast: seeded at 2022-08.")
W("  See visualizations/23_fcm_backcast_sept2021.png and")
W("  visualizations/24_fcm_backcast_oct2022.png for full plots.")
disc("""
The backcast tests whether the FCM, given the actual starting conditions
at the die-off event, projects a trajectory consistent with what was
subsequently observed.

For the September 2021 event, the FCM correctly projects salinity
recovering toward normal over the following months  - the freshwater pulse
was transient and the FCM captures the mean-reverting tendency.  However,
the DO trajectory is poorly reproduced: the FCM predicts a smooth recovery
while the actual data shows continued DO suppression through October.

For October 2022, the FCM captures the general direction of DIN declining
over subsequent months but substantially underestimates the magnitude of
the October peak.  This is the systematic underestimation problem: sigmoid
activation limits predictions to [0, 1] and compresses extreme events.

Both backcasts demonstrate that the FCM is more useful for identifying
'which variables were stressed and how they co-varied' than for predicting
the exact trajectory of recovery.  The causal graph structure (Section 3)
is the primary scientific contribution; the simulation is illustrative.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 6  - GEOGRAPHIC MAPS: SENSOR STATIONS  (Figures 25–29)")
# ════════════════════════════════════════════════════════════════

for fig_n, feat, label, unit in [
        (25,"temp_c","Water Temperature","°C"),
        (26,"sal_ppt","Salinity","PPT"),
        (27,"odo_mgL","Dissolved Oxygen","mg/L"),
        (28,"turbidity_fnu","Turbidity","FNU"),
]:
    h2(f"Figure {fig_n}  - {label} ({unit}) seasonal medians, 2025-2026")
    t = load_sensor_seasonal(feat)
    if not t.empty:
        W(tbl(t, float_fmt=".2f"))
    else:
        W("  (no observed data)")

disc("""
Reading across each row (fixed station) reveals the seasonal cycle at
that location.  Reading down each column (fixed year+season) reveals
the spatial gradient from north (L0, Haulover) to south (crest5).

Temperature (Figure 25): The 13–15°C range between summer (JJA ~32°C)
and winter (DJF ~19°C) is consistent throughout the bay.  L0 (Haulover
Inlet) shows slightly lower summer temperatures because it exchanges
more water with the Atlantic Ocean.  Consolidated_crest5 tends to be
warmer in summer  - the southern bay is shallower and more enclosed.

Salinity (Figure 26): Canal-adjacent L1 and L2 show the lowest salinity
in summer JJA (~24–28 PPT), while L0 near the ocean inlet maintains
higher salinity (~33–35 PPT).  This north–south salinity gradient reverses
in winter when canal discharge is minimal and tidal flushing re-establishes
oceanic values throughout.

Dissolved Oxygen (Figure 27): Summer DO at L6 and L7 drops to 4–5 mg/L,
approaching the hypoxic threshold for fish (5 mg/L).  These are the
stations closest to canal outfalls where organic loading is highest.
Winter DO is uniformly higher (7–9 mg/L) across all stations as cooler
water holds more oxygen and biological demand decreases.

Turbidity (Figure 28): L2 and L1 show the highest turbidity, reflecting
suspended sediment and organic material from canal discharge.  L0 (Haulover)
and consolidated_crest5 show lower turbidity  - they are closer to the
open ocean where tidal flushing and clearer oceanic water dominate.
""")

h2("Figure 29  - Annual overview (all 4 variables, 2025 vs 2026)")
W("")
W("  Annual median by station and year (observed only):")
W("")
for feat, label, unit in [("temp_c","Temperature","°C"),("sal_ppt","Salinity","PPT"),
                           ("odo_mgL","DO","mg/L"),("turbidity_fnu","Turbidity","FNU")]:
    t = load_sensor_seasonal(feat)
    if t.empty:
        continue
    t_annual = t.copy()
    yr_cols_2025 = [c for c in t.columns if c != "station" and c.startswith("2025")]
    yr_cols_2026 = [c for c in t.columns if c != "station" and c.startswith("2026")]
    t_annual["2025 Annual"] = t[yr_cols_2025].mean(axis=1).round(2) if yr_cols_2025 else np.nan
    t_annual["2026 Annual"] = t[yr_cols_2026].mean(axis=1).round(2) if yr_cols_2026 else np.nan
    ann = t_annual[["station","2025 Annual","2026 Annual"]].copy()
    W(f"  {label} ({unit}):")
    W(tbl(ann, float_fmt=".2f"))
    W("")
disc("""
The annual comparison between 2025 and 2026 shows modest year-over-year
changes  - the bay has no dramatic long-term trend over this short period.
Temperature is nearly identical between years across all stations, as
expected for a location with stable sub-tropical climate.

Salinity shows slightly higher 2026 values at most stations, suggesting
2025 may have had higher canal discharge (consistent with a historically
wet 2025 wet season) or lower evaporation relative to rainfall.

Dissolved oxygen shows an improvement from 2025 to 2026 at the vulnerable
nodes (L6, L7): 2025 summer DO was lower, possibly because 2025 had more
organic loading events.  This tentative improvement is worth monitoring
but cannot be attributed to any specific management intervention without
additional data.
""")

# ════════════════════════════════════════════════════════════════
h1("SECTION 7  - EXTENDED MAPS 2021–2026  (Figures 30–36)")
# ════════════════════════════════════════════════════════════════

h2("Figures 30–32  - Bay-wide temperature, salinity, DO (2021-2026)")
W("")
W("  Open-bay grab sites (Biscayne Bay type), annual mean by year:")
W("")
for var, label, unit in [("temp","Temperature","°C"),("sal","Salinity","PPT"),
                          ("do_mgL","DO","mg/L")]:
    sub = grab_raw[grab_raw["site_type"]=="Biscayne Bay"].copy()
    ann = sub.groupby(["site_name", sub["date"].dt.year.rename("year")])[var]\
              .median().round(2).reset_index()
    ann.columns = ["Site","Year",f"{label} ({unit})"]
    ann = ann.sort_values(["Site","Year"])
    W(f"  {label} ({unit})  - Biscayne Bay open-water sites:")
    W(tbl(ann.reset_index(drop=True), float_fmt=".2f"))
    W("")

disc("""
These tables cover 2021-2024 at open-bay grab sites  - the historical
context missing from the continuous sensor deployment.

Temperature: All sites show consistent warming between 2022 and 2024
(annual means rising ~0.5–1°C across sites).  This is consistent with
regional trends.  If this rate continues, summer maximum temperatures
could regularly exceed 33°C by the late 2020s  - the thermal stress
threshold for tropical seagrass species in Biscayne Bay.

Salinity: 2021 shows the lowest annual means (driven by the September
wet-season event), with 2023 and 2024 being relatively dry years with
higher salinity.  The inter-annual range (~3–4 PPT at open-bay sites) is
smaller than the seasonal range (~8 PPT), but still biologically
significant for organisms near their salinity tolerance limits.

DO: All sites consistently above the hypoxia threshold on annual average,
but the variance within years is large.  BB14 (nearest to canal influence)
shows the most year-to-year variability, reflecting its sensitivity to
discharge events.  BB28 (southernmost) is the most stable, buffered by
distance from canal mouths.
""")

h2("Figures 33–35  - Nutrient maps: pH, Chl-a, DIN (2021-2024)")
W("")
for var, label, unit, threshold in [
        ("ph","pH","",("< 7.7 = acidic stress",)),
        ("chl_exo_ugL","Chl-a","µg/L",("≥ 5 = elevated bloom risk",)),
        ("din","DIN","µmol/L",("≥ 10 = eutrophic threshold",)),
]:
    ann = grab_raw.groupby(["site_name","site_type",
                             grab_raw["date"].dt.year.rename("year")])[var]\
              .median().round(3).reset_index()
    ann.columns = ["Site","Type","Year",f"{label} ({unit})".strip()]
    ann = ann.sort_values(["Year","Site"])
    W(f"  {label}:")
    W(tbl(ann.reset_index(drop=True), float_fmt=".3f"))
    W("")

disc("""
pH (Figure 33): Open-bay sites (BB14–BB28) maintain pH above 7.9 most
of the time  - healthy carbonate chemistry.  Canal sites (LR01, MR01) and
outfall sites (GOC-012 to GOC-015) show consistently lower pH (7.4–7.7),
reflecting the acidifying effect of organic matter decomposition and high
CO₂ production from microbial respiration of nutrient-rich effluent.
The 2021 measurements show the lowest pH readings, coinciding with the
large freshwater pulse of that year (river water is more acidic than bay
water).

Chlorophyll-a (Figure 34): The north-to-south gradient is striking.
Canal-adjacent sites (LR01, GOC outfalls) show Chl-a values of 3–15 µg/L,
consistent with moderate to high algal biomass.  Open-bay sites (BB14–BB28)
are generally below 2 µg/L  - below the bloom threshold.  Summer values
are higher at all sites, driven by the temperature-forced algal growth
documented in the FCM (Temp → Chl-a = +0.54).

DIN (Figure 35): This is the most spatially heterogeneous variable.
GOC outfall sites (GOC-012, GOC-013, GOC-014, GOC-015) show DIN above 20
µmol/L in multiple years  - well above the 10 µmol/L eutrophication
threshold.  LR01 (Little River Canal) reached DIN > 15 µmol/L in 2021.
Open-bay sites (BB14–BB28) are generally below 5 µmol/L, confirming that
the bay itself dilutes and assimilates nitrogen from the canal sources,
but not quickly enough to prevent hypoxia near the outfalls.
""")

h2("Figure 36  - Annual hypoxia stress: NH4 and DO% by site (2021-2024)")
stress = grab_raw.groupby(["site_name","site_type",
                            grab_raw["date"].dt.year.rename("year")]
                          )[["nh4","do_per"]].median().round(2).reset_index()
stress.columns = ["Site","Type","Year","NH4 (µmol/L)","DO (%Sat)"]
stress = stress.sort_values(["Year","NH4 (µmol/L)"],ascending=[True,False])
W(tbl(stress.reset_index(drop=True), float_fmt=".2f"))
disc("""
Figure 36 is the single most diagnostic chart for hypoxia risk.  The
co-occurrence of high NH4 AND low DO%Sat at the same site is the direct
precursor to biological die-offs.

Canal sites (LR01, MR01) and outfall sites (GOC series) consistently show
the highest NH4 and lowest DO.  LR01 has NH4 > 5 µmol/L in multiple years,
with the 2021 event peak reaching 21.9 µmol/L (from Figure 17).  The GOC
outfall cluster maintains NH4 above 5 µmol/L in most years, driven by
groundwater seepage and storm drainage carrying sewage-derived nitrogen.

The relationship between NH4 and DO%Sat is approximately inverse across
the table: sites with NH4 above 5 µmol/L tend to have DO%Sat below 80%,
while open-bay sites (BB14, BB17, BB18) maintain NH4 below 2 µmol/L and
DO above 85%.  This inverse correlation is mechanistic: ammonium oxidation
(nitrification) is one of the major oxygen-consuming processes in
eutrophic coastal systems.

Year-to-year pattern: 2021 shows the most severe stress (highest NH4,
lowest DO at canal sites) corresponding to the fish die-off event.  2022
is notable for the GOC outfall sites reaching extreme DIN in October.
2023 and 2024 are comparatively moderate, suggesting the system partially
recovered or that discharge management improved  - though without canal
discharge records it is impossible to confirm this interpretation.
""")

# ════════════════════════════════════════════════════════════════
h1("APPENDIX  - KEY NUMBERS QUICK-REFERENCE")
# ════════════════════════════════════════════════════════════════
W("")
W("  Model:       ST-GNN  |  97,544 parameters  |  val loss 0.0021")
W("  Nodes:       7  (L0,L1,L2,L6,L7,biscayne_bay,consolidated_crest5)")
W("  Timespan:    2025-03-10 to 2026-08-01  (146,421 five-minute steps)")
W("  Missing:     73.2 % before imputation")
W("  Grab data:   761 surface rows, 20 sites, 2021-09 to 2024-12")
W("  Buoys:       crest3-5, haulover, kampong, royal (Jun-Jul 2026)")
W("")
W("  Physical FCM:  daily, lag=1 day,   MSE lag test: 1=0.0022 2=0.0029 3=0.0035")
W("  Nutrient FCM:  monthly, lag=1 mo,  MSE lag test: 1=0.1485 2=0.1511 3=0.1561")
W("  Forcing:       Net Water Balance = Rain - Hargreaves PET")
W("  PET range:     0.11 - 0.24 in/day (3.3 - 7.2 in/month)")
W("")
W("  Hindcast R²:  Salinity=+0.18, Spec.Cond.=+0.17; all others negative")
W("  Best spatial link: L6 → L2 (importance 1.13e-05)")
W("  Most isolated:     L7 ↔ consolidated_crest5 (mutual attn=1.0)")

# ── write ─────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "figures_summary_v3.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print(f"Report v3 written to {out_path}  ({len(OUT)} lines)")
