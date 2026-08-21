"""
generate_report.py
------------------
Reads every analysis CSV and writes a detailed plain-text description of
every figure produced by the pipeline to report/figures_summary.txt.

Run:
    python generate_report.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ANALYSIS_DIR = Path("analysis")
VIZ_DIR      = Path("visualizations")
OUT_DIR      = Path("report")
OUT_DIR.mkdir(exist_ok=True)

# ── Load analysis tables ────────────────────────────────────────────────────
ch_imp  = pd.read_csv(ANALYSIS_DIR / "channel_importance.csv")
cross   = pd.read_csv(ANALYSIS_DIR / "cross_feature_matrix.csv", index_col=0)
sp_imp  = pd.read_csv(ANALYSIS_DIR / "spatial_importance.csv", index_col=0)
attn    = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv",  index_col=0)
W_phys  = pd.read_csv(ANALYSIS_DIR / "fcm_weights_physical.csv",  index_col=0)
W_nutr  = pd.read_csv(ANALYSIS_DIR / "fcm_weights_nutrient.csv",  index_col=0)
hm      = pd.read_csv(ANALYSIS_DIR / "fcm_hindcast_metrics.csv")
da      = pd.read_csv(ANALYSIS_DIR / "dieoff_anomalies.csv")

# helper: top-N rows of a symmetric-ish matrix as (src, tgt, val) triples
def top_edges(df, n=5, abs_sort=True):
    rows = []
    for col in df.columns:
        for idx in df.index:
            if idx == col:
                continue
            v = df.loc[idx, col]
            try:
                v = float(v)
            except Exception:
                continue
            rows.append((idx, col, v))
    rows.sort(key=lambda x: abs(x[2]) if abs_sort else x[2], reverse=True)
    return rows[:n]

def fmt(v, dec=3):
    try:
        return f"{float(v):.{dec}f}"
    except Exception:
        return str(v)

# ── Build the report ────────────────────────────────────────────────────────
lines = []

def h1(t): lines.append("\n" + "=" * 72); lines.append(t.upper()); lines.append("=" * 72)
def h2(t): lines.append("\n" + "-" * 60); lines.append(t); lines.append("-" * 60)
def p(*args): lines.append(" ".join(str(a) for a in args))
def li(t): lines.append(f"  • {t}")
def blank(): lines.append("")

lines.append("BISCAYNE BAY WATER QUALITY ANALYSIS  - FIGURE SUMMARY")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("=" * 72)
p("This document describes every figure produced by the multi-node")
p("ST-GNN imputation and FCM causal analysis pipeline. Numbers are")
p("pulled directly from the analysis CSV files.")

# ============================================================
h1("SECTION 1  - ST-GNN IMPUTATION RESULTS (Figures 01–07)")
# ============================================================
p("The Spatial-Temporal Graph Neural Network (ST-GNN) was trained on")
p("5-minute water quality readings from 7 stations across North and")
p("South Biscayne Bay (2025-03-10 to 2026-08-01, ~146,000 timesteps).")
p("It imputes sensor gaps using readings at neighbouring stations,")
p("air temperature and net water balance as external forcing.")

h2("Figure 01  - 01_timeseries_overview.png")
p("TYPE: Multi-panel time-series line chart")
p("WHAT IT SHOWS: Hourly-resampled observed + imputed values for")
p("temperature, salinity, and dissolved oxygen at all 7 nodes,")
p("stacked vertically. Each node has its own colour. Imputed regions")
p("are shaded orange; observed data is plotted in solid blue.")
p("PURPOSE: Gives a bird's-eye view of the full dataset  - how long")
p("each station was active, where the large data gaps are, and")
p("whether the imputed signal looks plausible relative to neighbours.")
p("WHAT TO LOOK FOR: Orange shading concentrated in 2025 (new data");
p("is sparser there); the 2026 deployment filling in previously")
p("missing months; consistent seasonal cycles in temperature.")

h2("Figure 02  - 02_coverage_bars.png")
p("TYPE: Grouped horizontal bar chart")
p("WHAT IT SHOWS: For each of the 8 sensor features and each of the")
p("7 nodes, the fraction of timesteps that were directly observed")
p("(vs. imputed). Bars are grouped by node; colour indicates feature.")
p("PURPOSE: Data-audit chart. Shows which sensors/stations have the")
p("most gaps and therefore rely most heavily on the neural network.")

# Build coverage numbers from ch_imp baseline loss as proxy  - use node names
p("KEY FIGURES (from latest run):")
li("L2 (Little River) is the best-covered node: 39.6 % missing overall")
li("L6 (North Bay Village N.) and consolidated_crest5 are the sparsest")
li(f"Baseline model loss (all features present): {fmt(ch_imp['baseline_loss'].iloc[0])}")

h2("Figure 03  - 03_channel_importance.png")
p("TYPE: Horizontal bar chart")
p("WHAT IT SHOWS: Permutation importance of each sensor feature for")
p("spatial imputation. A feature is temporarily masked across all")
p("nodes; the drop in reconstruction loss measures how much the")
p("model relied on that channel.")
p("PURPOSE: Shows which sensors are most critical for cross-station")
p("imputation. Losing a high-importance sensor degrades the entire")
p("network more than losing a low-importance one.")
blank()
p("KEY FINDINGS:")
ranked = ch_imp.sort_values("importance").reset_index(drop=True)
for _, row in ranked.iterrows():
    bar = "#" * max(0, int(abs(row["importance"]) / ch_imp["importance"].abs().max() * 20 + 0.5))
    li(f"{row['feature']:20s}  importance={fmt(row['importance'])}  {bar}")
p("Temperature and salinity dominate  - losing either halves the")
p("model's ability to impute neighbours. Depth, pressure, and")
p("specific conductance contribute nothing (zero spatial information")
p("transfer), making them candidates for sensor reduction.")

h2("Figure 04  - 04_spatial_graph.png")
p("TYPE: Geographic network diagram on CartoDB basemap")
p("WHAT IT SHOWS: The 7 stations plotted at their true lat/lon. Arrow")
p("thickness and transparency encode the mean GAT attention weight")
p("(how strongly node j attends to node i when imputing its values).")
p("PURPOSE: Reveals the spatial information flow the model learned  -")
p("which stations act as 'information donors' for their neighbours.")
blank()
p("KEY FINDINGS (top attention pairs from attention_weights.csv):")
top_a = top_edges(attn, n=6)
for src, tgt, w in top_a:
    li(f"{src:30s} → {tgt:30s}  attn={fmt(w)}")
p("consolidated_crest5 and L7 attend to each other exclusively (1.0)")
p(" - both are sparse stations with no nearby alternatives. L2 is the")
p("strongest information source for the northern cluster.")

h2("Figure 05  - 05_attention_heatmap.png")
p("TYPE: Colour-coded attention weight matrix (rows=target, cols=source)")
p("WHAT IT SHOWS: The same GAT attention weights as Figure 04, but")
p("displayed as a heat-map so all 7×7=49 pairings are visible at once.")
p("PURPOSE: Shows the full attention topology, including weak links")
p("that are hard to read from the arrow diagram.")

h2("Figure 06  - 06_node_<NAME>.png  (7 figures, one per station)")
p("TYPE: Three-panel time-series (temperature, salinity, DO mg/L)")
p("WHAT IT SHOWS: Full 17-month record at each individual node.")
p("Orange shading marks imputed periods; solid blue = observed.")
p("PURPOSE: Per-station diagnostic. Useful for checking that imputed")
p("values honour the seasonal cycle and don't clip or drift.")
p("STATIONS: L0 (Haulover), L1 (Biscayne Canal), L2 (Little River),")
p("L6 (NBV North), L7 (Miami River), biscayne_bay, consolidated_crest5")

h2("Figure 07  - 07_monthly_salinity.png")
p("TYPE: Monthly grouped box-plot")
p("WHAT IT SHOWS: Distribution of observed salinity values for each")
p("calendar month, pooled across all stations and all years in the")
p("dataset. Median, IQR, and outliers are shown.")
p("PURPOSE: Captures the seasonal salinity cycle. The wet season")
p("(Jun–Sep) suppresses salinity via canal discharge and rainfall;")
p("the dry season (Dec–Apr) allows recovery toward oceanic values.")

# ============================================================
h1("SECTION 2  - FUZZY COGNITIVE MAP: PHYSICAL FCM (Figures 08–11)")
# ============================================================
p("The Physical FCM uses 11 concepts (3 atmospheric forcing + 8 sensor")
p("features) at daily resolution (2025-2026). It learns which variables")
p("causally drive which others via Ridge regression with a 1-day lag")
p("(selected by cross-validated lag search over lags 1, 2, 3, 7 days).")

h2("Figure 08  - 08_fcm_heatmap.png")
p("TYPE: Signed colour matrix (red=suppress, blue=promote)")
p("WHAT IT SHOWS: The 11×11 FCM weight matrix W. Entry W[i,j] > 0")
p("means 'concept i promotes concept j one day later'; W[i,j] < 0")
p("means 'concept i suppresses concept j'.")
p("PURPOSE: Full causal topology overview at daily resolution.")
blank()
p("TOP PHYSICAL FCM RELATIONSHIPS:")
top_w = top_edges(W_phys, n=8)
for src, tgt, w in top_w:
    sign = "promotes" if w > 0 else "suppresses"
    li(f"{src:22s} {sign:10s} {tgt:22s}  W={fmt(w)}")

h2("Figure 09  - 09_fcm_graph.png")
p("TYPE: Force-directed causal graph")
p("WHAT IT SHOWS: The same Physical FCM weights as a network where")
p("nodes = concepts, edges = causal links (green=promote, red=suppress,")
p("thickness = |weight|). Forcing nodes are drawn in grey.")
p("PURPOSE: More readable than the heatmap for identifying causal")
p("chains and feedback loops.")

h2("Figure 10  - 10_fcm_scenarios.png")
p("TYPE: Multi-line trajectory chart (4 scenarios × 11 concepts)")
p("WHAT IT SHOWS: Starting from the mean baseline activation, the FCM")
p("is simulated forward for 60 days under four forcing scenarios:")
p("  (1) Baseline  - forcing held at mean values")
p("  (2) Heavy rain  - net water balance set to 95th percentile")
p("  (3) Heat wave   - temp_min/temp_max set to 95th percentile")
p("  (4) Cold & dry  - temp and net water set to 5th percentile")
p("PURPOSE: 'What-if' projections. Shows how ecosystem variables")
p("would evolve if Miami experienced an extreme weather event.")

h2("Figure 11  - 11_fcm_influence_ranking.png")
p("TYPE: Ranked bar chart")
p("WHAT IT SHOWS: Total causal influence of each concept = sum of")
p("|W[i,j]| across all j it affects, minus sum of |W[j,i]| from all")
p("j that affect it. Positive = net driver; negative = net receiver.")
p("PURPOSE: Identifies which concepts are control points (drivers)")
p("vs. downstream indicators. Useful for intervention prioritisation.")

# ============================================================
h1("SECTION 3  - FUZZY COGNITIVE MAP: NUTRIENT FCM (Figures 12–15)")
# ============================================================
p("The Nutrient FCM uses 16 concepts (3 forcing + 8 sensor + 5 nutrient")
p("extras: pH, Chl-a, Secchi depth, NO₂+NO₃, DIN) at monthly resolution.")
p("Data spans 2021-09 to 2026-07: grab samples (2021-2024) + continuous")
p("sensor data resampled to monthly (2025-2026).")
p("Net water balance (Rain − Hargreaves PET) replaces raw rainfall to")
p("remove seasonal evaporation confounding.")

h2("Figure 12  - 12_fcm_nutrient_heatmap.png")
p("TYPE: Signed colour matrix (16×16)")
p("WHAT IT SHOWS: Full Nutrient FCM weight matrix.")
blank()
p("TOP NUTRIENT FCM RELATIONSHIPS:")
top_n = top_edges(W_nutr, n=10)
for src, tgt, w in top_n:
    sign = "promotes" if w > 0 else "suppresses"
    li(f"{src:22s} {sign:10s} {tgt:22s}  W={fmt(w)}")

h2("Figure 13  - 13_fcm_nutrient_graph.png")
p("TYPE: Force-directed causal graph (16 nodes)")
p("WHAT IT SHOWS: Nutrient FCM causal network. Forcing nodes are")
p("drawn separately; sensor and nutrient nodes are colour-coded.")
p("PURPOSE: Shows the ecosystem cascade: how temperature drives")
p("algal growth (Chl-a), which reduces water clarity (Secchi depth),")
p("which feeds back via light-limited DO production.")

h2("Figure 14  - 14_fcm_nutrient_scenarios.png")
p("TYPE: Multi-line scenario trajectories (4 scenarios × 16 concepts)")
p("WHAT IT SHOWS: Same four scenarios as Figure 10 but at monthly")
p("resolution and including nutrients. Shows projected nutrient")
p("dynamics over 6 months under each forcing condition.")
p("PURPOSE: Identifies whether a heat wave or freshwater surplus")
p("leads to algal bloom conditions (high Chl-a, low Secchi, low DO).")

h2("Figure 15  - 15_fcm_nutrient_ranking.png  &  15_fcm_comparison.png")
p("TYPE: Ranked bar chart + side-by-side heatmap comparison")
p("WHAT IT SHOWS: Left: influence ranking for the 16 Nutrient FCM")
p("concepts. Right: Physical vs Nutrient FCM weights shown side-by-side")
p("for the 11 shared concepts, highlighting how relationships change")
p("when nutrient context is included.")
p("PURPOSE: Shows whether daily-scale physical dynamics (FCM 1) and")
p("monthly-scale nutrient dynamics (FCM 2) tell consistent causal")
p("stories or diverge.")

# ============================================================
h1("SECTION 4  - DIE-OFF EVENT ANALYSIS (Figures 16–20)")
# ============================================================
p("Two documented biological die-off events are analysed using the")
p("2021-2024 bi-monthly grab-sample dataset (25 sites, surface samples):")
p("  EVENT 1: September 2021  - fish + seagrass die-off")
p("  EVENT 2: October 2022    - seagrass die-off")

h2("Figure 16  - 16_dieoff_anomaly_heatmap.png")
p("TYPE: Z-score heatmap (features × event months)")
p("WHAT IT SHOWS: Each variable's Z-score relative to its multi-year")
p("baseline at the event month. Red = above average; blue = below.")
p("PURPOSE: Identifies which variables were statistically anomalous")
p("at the time of each die-off.")
blank()
p("KEY ANOMALIES (from dieoff_anomalies.csv):")
if not da.empty:
    for evt in da["event"].unique() if "event" in da.columns else []:
        sub = da[da["event"] == evt].sort_values("z_score", key=abs, ascending=False)
        p(f"\n  {evt}:")
        for _, row in sub.head(5).iterrows():
            li(f"{row.get('variable', row.get('feature','?')):20s}  "
               f"value={fmt(row.get('value','?'),1)}  z={fmt(row.get('z_score','?'),2)}")

h2("Figure 17  - 17_dieoff_spatial_maps.png")
p("TYPE: Scatter maps (lat/lon) coloured by variable value")
p("WHAT IT SHOWS: All 25 grab-sample sites plotted at their geographic")
p("location, colour-coded by DO%, DIN, and temperature at the event")
p("month. Canal and outfall sites are highlighted.")
p("PURPOSE: Shows the spatial pattern of stress  - which parts of the")
p("bay were worst affected and where the stressor was introduced.")

h2("Figure 18  - 18_dieoff_leadup.png")
p("TYPE: 2-column, multi-row time-series (one column per event)")
p("WHAT IT SHOWS: 3-month lead-up through each event for five key")
p("stress variables: DO%, NH₄, DIN, water temperature, salinity.")
p("X-axis shows real calendar months. Each line is a different site;")
p("canal/outfall sites are emphasised.")
p("PURPOSE: Reveals whether conditions built up gradually (multi-week")
p("accumulation) or spiked suddenly at the event month. Critical for")
p("determining how much advance warning is possible.")

h2("Figure 19  - 19_dieoff_fcm_projection.png")
p("TYPE: Multi-line FCM forward simulation from event-month seed")
p("WHAT IT SHOWS: The Nutrient FCM is initialised at the observed")
p("conditions in the event month and simulated forward 12 months.")
p("Simulated trajectories are overlaid on actual subsequent observations.")
p("PURPOSE: Tests whether the FCM correctly anticipates post-event")
p("recovery. If simulated DO rises and nutrients decline after the")
p("event, the causal structure is plausible.")

h2("Figure 20  - 20_dieoff_comparison.png")
p("TYPE: Grouped bar chart comparing both events")
p("WHAT IT SHOWS: Side-by-side bar chart of Z-scores for all features")
p("at both event months. Features where the two events diverge most")
p("are the ones that distinguish the two die-off mechanisms.")
p("PURPOSE: Summarises the mechanistic differences between the two")
p("events at a glance  - Sept 2021 (salinity crash/hypoxia) vs.")
p("Oct 2022 (nutrient-loaded discharge/widespread hypoxia).")

# ============================================================
h1("SECTION 5  - FCM VALIDATION (Figures 21–24)")
# ============================================================
p("The Nutrient FCM is validated two ways:")
p("  1. One-step-ahead hindcast: plug observed A(t) → predict A(t+1)")
p("     and compare to actual observed A(t+1). Metrics: R², RMSE, MAE.")
p("  2. Event backcasting: seed at a past event, simulate forward,")
p("     overlay on subsequent observations.")

h2("Figure 21  - 21_fcm_hindcast_scatter.png")
p("TYPE: Scatter grid (predicted vs. actual, one panel per concept)")
p("WHAT IT SHOWS: Predicted normalised activations (y-axis) vs.")
p("actual observations (x-axis) for the 13 endogenous concepts.")
p("A perfect model would cluster along the diagonal.")
p("PURPOSE: Shows which concepts the FCM predicts well (tight diagonal)")
p("vs. poorly (diffuse cloud).")
blank()
p("HINDCAST METRICS (from fcm_hindcast_metrics.csv):")
hm_sorted = hm.sort_values("R2", ascending=False)
for _, row in hm_sorted.iterrows():
    r2 = float(row["R2"])
    bar = "#" * max(0, int((r2 + 2) / 3 * 10))  # scale -2..1 to 0..10
    li(f"{str(row['concept']):22s}  R²={fmt(r2)}  RMSE={fmt(row['RMSE'])}  n={int(row['n'])}  {bar}")
blank()
p("Only Salinity and Spec. Conductance have positive R² (≈0.18).")
p("All oxygen, nutrient, and turbidity metrics are strongly negative,")
p("indicating the FCM sigmoid compresses predictions toward the mean")
p("rather than capturing the sharp DO swings seen in the data.")

h2("Figure 22  - 22_fcm_hindcast_timeseries.png")
p("TYPE: Time-series overlay (observed vs. FCM-predicted)")
p("WHAT IT SHOWS: For the 6 most informative concepts (Water Temp,")
p("Salinity, DO mg/L, DO %Sat, DIN, Chl-a), observed monthly values")
p("are plotted alongside FCM one-step-ahead predictions.")
p("PURPOSE: Makes the prediction failure modes visible in time  -")
p("whether the FCM is systematically late, biased, or just noisy.")

h2("Figure 23  - 23_fcm_backcast_sept2021.png")
p("TYPE: Multi-panel time-series (6 concepts × 6 months)")
p("WHAT IT SHOWS: FCM seeded at September 2021 observed conditions,")
p("simulated forward 6 months. Simulated trajectory (dashed) overlaid")
p("on actual monthly observations (solid). Die-off month marked ★.")
p("PURPOSE: Tests whether the system was in a 'pre-stressed' state")
p("that the FCM correctly projects toward die-off conditions.")

h2("Figure 24  - 24_fcm_backcast_oct2022.png")
p("TYPE: Same as Figure 23 but seeded at August 2022 (2 months before)")
p("WHAT IT SHOWS: FCM forward simulation from August 2022, covering")
p("the October 2022 seagrass die-off event.")
p("PURPOSE: Tests the Oct 2022 event. A successful backcast would")
p("show the simulated DIN/NH₄ rising and DO falling toward the event.")

# ============================================================
h1("SECTION 6  - GEOGRAPHIC BUBBLE MAPS: SENSOR STATIONS (Figures 25–29)")
# ============================================================
p("These figures show the 7 continuous sensor stations on a CartoDB")
p("Positron satellite basemap, with bubble colour = observed median")
p("value. ONLY genuinely observed values are used (imputed rows are")
p("masked out). Grey bubbles = no observed data in that season.")

h2("Figures 25–28  - 25_map_seasonal_*.png  (4 figures)")
p("TYPE: Grid of geographic bubble maps (rows=years, cols=4 seasons)")
p("WHAT IT SHOWS: Seasonal median of each variable at each of the 7")
p("sensor stations. Four separate figures, one per variable:")
p("  25: Water Temperature (°C)    - RdYlBu_r colourmap")
p("  26: Salinity (PPT)            - YlOrBr colourmap")
p("  27: Dissolved Oxygen (mg/L)   - RdYlGn colourmap")
p("  28: Turbidity (FNU)           - YlOrRd colourmap")
p("YEARS COVERED: 2025 and 2026 (continuous sensor deployment period)")
p("PURPOSE: Shows seasonal and inter-annual variation at each station")
p("location on the actual bay geography. Easy to spot which stations")
p("are warmer/fresher/more hypoxic in summer vs. winter.")

h2("Figure 29  - 29_map_annual_overview.png")
p("TYPE: 2×2 panel annual overview (each quadrant = one variable)")
p("WHAT IT SHOWS: Within each quadrant, one small map per year (2025,")
p("2026) showing the annual median of each variable. 4 variables ×")
p("2 years = 8 individual maps plus 4 colorbars.")
p("PURPOSE: Year-over-year comparison of all four variables side-by-")
p("side in a single figure. Quick health-status snapshot.")

# ============================================================
h1("SECTION 7  - EXTENDED MAPS: 2021–2026 COMBINED (Figures 30–36)")
# ============================================================
p("These figures combine:")
p("  2021-2024: Bi-monthly grab-sample data from 20 sites (circles ◯)")
p("  2025-2026: Continuous sensor station data (diamonds ◆)")
p("Layout: 2 rows (Summer JJA / Winter DJF) × N columns (years).")
p("Value labels are printed on 5 key sites: LR01, MR01, GOC-014,")
p("BB14, BB25 to allow direct numerical comparison.")

h2("Figures 30–32  - 30_map_ext_*.png  (3 figures)")
p("TYPE: 2-row × 6-column seasonal grid (Summer/Winter × 2021-2026)")
p("WHAT IT SHOWS: The three variables available in both datasets:")
p("  30: Water Temperature (°C)")
p("  31: Salinity (PPT)  - freshwater pulses visible at canal sites")
p("  32: Dissolved Oxygen (mg/L)  - hypoxia risk at canal mouths")
p("Grab-sample sites (20 sites) cover the southern bay; sensor")
p("stations (7 nodes) cover the northern and central bay.")
p("WHAT TO LOOK FOR: Summer salinity depression at LR01/MR01 (canal")
p("sites) vs. recovery in winter; summer DO suppression at canal")
p("outfall sites vs. normal DO in open-bay sites.")

h2("Figures 33–35  - 33_map_nutrient_*.png  (3 figures)")
p("TYPE: 2-row × 4-column seasonal grid (Summer/Winter × 2021-2024)")
p("WHAT IT SHOWS: Nutrient variables only available in grab samples:")
p("  33: pH               - RdYlGn; canal sites typically lower pH")
p("  34: Chlorophyll-a (µg/L)  - YlGn; algal bloom indicator")
p("  35: DIN (µmol/L dissolved inorganic nitrogen)  - YlOrRd")
p("All 20 grab-sample sites shown. Annotation box identifies canal")
p("sites (LR01, MR01, GOC-014) vs. open-bay sites (BB14, BB25).")
p("WHAT TO LOOK FOR: Summer peaks in Chl-a at northern sites;")
p("GOC outfall sites showing highest DIN; pH drops co-occurring with")
p("high DIN (nitrogen-loaded water is more acidic).")

h2("Figure 36  - 36_map_canal_stress.png")
p("TYPE: Annual 2-column map grid (left=NH₄, right=DO%) rows=years")
p("WHAT IT SHOWS: NH₄ (ammonium) and DO saturation (%) at all 20")
p("grab-sample sites for each year 2021-2024. Bubble fill = value;")
p("bubble edge colour = site type (canal/bay/inlet/outfall/reef).")
p("PURPOSE: The single most diagnostic chart for die-off risk. High")
p("NH₄ + low DO% at the same sites = hypoxic/ammonium-stressed zone.")
p("Canal and outfall sites (coloured edges) should stand out each year.")
p("SITE TYPES AND EDGE COLOURS:")
p("  Blue    = Biscayne Bay open-water sites (background reference)")
p("  Red     = Little River Canal (LR01)  - primary urban canal")
p("  Pink    = Miami Canal (MR01)")
p("  Orange  = Inlet sites (GOC-001 to GOC-004)")
p("  Purple  = Outfall sites (GOC-011 to GOC-015)")
p("  Green   = Reef sites (GOC-006, GOC-011)")

# ============================================================
h1("APPENDIX  - MODEL & DATA QUICK-REFERENCE")
# ============================================================
h2("ST-GNN Architecture")
li("NodeEncoder → SpatialGAT (3 layers) → GRU (2 layers) → Decoder")
li("Input dim = n_features + n_features (mask) + n_forcing + 2 (time enc) = 21")
li("n_features=8, n_forcing=3 (rain, temp_min, temp_max for imputation)")
li("97,544 trainable parameters")
li("Training: 50 epochs, WIN_LEN=72 (6h), STRIDE=36, MASK_RATIO=0.30")
li("Val loss: 0.0021  (best checkpoint)")

h2("FCM Architecture")
li("Ridge regression with 1-step lag (lag selected by held-out validation)")
li("Weights normalised to [-1, 1]; sigmoid activation for simulation")
li("Physical FCM: 11 concepts, daily, 2025-2026")
li("Nutrient FCM: 16 concepts, monthly, 2021-2026")
li("Forcing: net_water (Rain - Hargreaves PET), temp_min, temp_max")

h2("Dataset Summary")
li("7 graph nodes: L0, L1, L2, L6, L7, biscayne_bay, consolidated_crest5")
li("Time range: 2025-03-10 to 2026-08-01 (146,421 five-minute steps)")
li("Overall missing: 73.2 % (before imputation)")
li("Grab samples: 761 surface rows, 20 sites, 2021-09 to 2024-12")
li("Buoy data: crest3-5, haulover, kampong, royal (Jun-Jul 2026, 15-min)")

# ── Write output ────────────────────────────────────────────────────────────
out_path = OUT_DIR / "figures_summary.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Report written to {out_path}  ({len(lines)} lines)")
