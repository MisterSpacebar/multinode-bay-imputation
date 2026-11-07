# Deck 1 (10-slide draft): From GNN imputation to FCM, and why forecasting is next

Audience: mixed environmental-science and data-science. Length: ~20 min. Follows from the longer [22-slide scaffold](deck1_gnn_to_fcm_scaffold.md); continues in [Deck 2](deck2_forecasting_scaffold.md).
Data window: continuous sensors Mar 2025-Sep 2026; monthly grab samples Sep 2021-Sep 2026. Paths are relative to this file. `[NEW]` = figure not built yet.

---

## 1. Can a patchy sensor network tell us about low oxygen in north Biscayne Bay?

**On slide**
- Question: what drives dissolved oxygen (DO) in the north bay, and can we see trouble coming?
- Three steps: fill the gaps (GNN) -> map relationships (FCM) -> test what the newest data and models can predict
- Data: 7 sensor nodes, Mar 2025-Sep 2026; monthly water-quality grabs 2021-2026

**Say**
- Two die-offs (Sep 2021, Oct 2022) motivated this. Our continuous sensors only began in 2025, so we ask what we can learn now and be upfront about what we can't.

**Visual:** [bay map](../report/canal_stress_map_v2.png) as a full-bleed background or right half.

---

## 2. The data problem: most of the record is missing

**On slide**
- Directly observed share ranges from 0% to 73% per node and feature
- Best-covered: L2 (~72% for temp/salinity/DO); worst: L6 (~12%), crest5 and bay (depth/pressure 0%)
- North bay has continuous sensors only since Mar 2025

**Say**
- Every downstream analysis inherits this. We keep an observed/imputed flag on every value so we can always score on real data only.

**Visual:** [coverage heatmap](../multinode_visuals/02_coverage_heatmap.png) (left, large). Optional: [NEW] relabel `consolidated_crest5` as "crest5".

---

## 3. The GNN: stations as a graph that fills each other's gaps

**On slide**
- Stations = nodes; distance = edges; attention learns who informs whom
- Encoder -> 3 attention layers -> GRU -> decoder; trained by masking real data and reconstructing it
- Learned: salinity (+0.058) and temperature (+0.036) carry the signal; spatial edges barely matter (~1e-4)
- Flag: L7-crest5 attend only to each other, a single point of failure

**Say**
- Keep the architecture to one visual; the point is the learned structure. Salinity and temperature are the bay-wide carrier signals; depth, pressure, and conductivity are local.

**Visuals:** left [GNN concept](../multinode_visuals/13_gnn_concept.png); right [channel importance](../multinode_visuals/12_channel_importance_clean.png). Backup: [architecture](../multinode_visuals/01_architecture_diagram.png), [attention network](../multinode_visuals/03_attention_network_annotated.png), [real gap before/after](../multinode_visuals/10_before_after_imputation.png).

---

## 4. Honest scorecard: where the GNN earns its keep

**On slide**
- Random 30% gaps: linear interpolation wins (DO 0.48 vs GNN 0.52 mg/L; temp 1.41 vs 1.95; salinity 2.41 vs 3.26)
- GNN beats climatology and spatial-mean baselines by a wide margin
- Real deployment gaps are weeks to months long (crest5 ~87% missing), where interpolation has nothing to anchor to
- Takeaway: use the GNN for structural gaps; use interpolation for short ones

**Say**
- Present this as a strength of the evaluation, not a failure. Beating a baseline is the only honest metric, and the metric has to match the use case.

**Visual:** [baseline overview](../multinode_visuals/15_baseline_overview.png) (single full slide; the bottom strip does the explaining).

---

## 5. The FCM: a transparent map of what moves together

**On slide**
- GNN reconstructs; the FCM shows relationships we can read
- Concepts = drivers/states; arrows = signed, weighted, lagged associations learned with ridge regression
- Two maps: daily physical (11 concepts), monthly nutrient (16 concepts, grab data to 2024)
- Weights are associations, not causal effects

**Say**
- This is a hypothesis map for scientists and a communication layer, built on the gap-free GNN output.

**Visual:** [data to FCM pipeline](../FCM/graphics/01_data_to_fcm.png).

---

## 6. What the FCM suggests about the ecology

**On slide**
- Air temp -> water temp (+0.52)
- Rain minus evaporation -> nitrite+nitrate (+0.42) and chlorophyll (+0.43)
- Water temp -> DO is weak and negative (-0.22): warmer water, less oxygen
- Chlorophyll -> Secchi depth (+0.46); DIN -> Secchi (-0.32)
- Event context: 2021 looks like freshwater dilution; 2022 like saline nutrient loading

**Say**
- Show only 8 concepts / 10 edges (of 150) for readability; a display filter, not a significance test. Event context is descriptive, not a validated mechanism.

**Visuals:** [ecology focus map](../FCM/graphics/04_ecology_focus_map.png) (main). Optional right/inset: [event context](../FCM/graphics/06_event_context.png). Backup: [full nutrient map](../FCM/graphics/03_nutrient_concept_map.png), [scenarios](../visualizations/10_fcm_scenarios.png) (hypothesis exploration only).

---

## 7. But the FCM cannot predict yet

**On slide**
- In-sample monthly hindcast R^2 is 0.17-0.84 (DO mg/L 0.77), mostly "this month predicts next"
- Held-out (last 17 months), persistence beats the FCM on all 5 testable concepts (DO RMSE 0.096 vs 0.160, scaled)
- Backcasts relax to a flat fixed point; Sep 2021 backcast is seeded with the event month
- Verdict: a hypothesis map, not an early-warning tool

**Say**
- Saying this plainly makes the rest of the deck credible. What the FCM needs: rolling-origin tests, provenance masks, matched lags, baselines.

**Visual:** [FCM vs persistence](../FCM/graphics/07_fcm_vs_persistence.png) (main); [validation scorecard](../FCM/graphics/05_validation_scorecard.png) (backup).

---

## 8. What the newest data show: a recurring low-oxygen season

**On slide**
- L1 canal below 4 mg/L for 22.2 h/day in Sep 2025 and 23.7 h/day in Sep 2026
- L2: 20-23 h/day through Dec 2025
- South reference sites are low in Sep 2026 too (L7 21 h/day, crest5 22 h/day), so not only a north-canal effect
- Caveat: from Jun 2026 L0, L2, L7, p2, crest5 also use buoy data; only L1 and L6 are like-for-like

**Say**
- Observed values only; imputed values excluded. Sep 2026 is a repeat of Sep 2025, not a new event. L0 dropping to ~0 h/day in 2026 likely reflects the move to the Haulover inlet buoy, not recovery.

**Visuals:** top [hypoxia heatmap](../visualizations/30_north_bay_hypoxia_heatmap.png); bottom/right [2025 vs 2026](../visualizations/31_north_bay_hypoxia_2025_vs_2026.png) (L1 is the clean comparison; call out the sensor-change nodes).

---

## 9. What this means for each audience

**Environmental science**
- Low oxygen is seasonal and recurring in the north bay (Sep-Dec); plan monitoring around it
- crest5 is the worst site (22-24 h/day <4 mg/L, 10-13 h/day <2 mg/L, Jul-Sep 2026)
- Candidate drivers to study: rain/freshwater, temperature, nutrients (hypotheses from the FCM)

**Data science / analytics**
- Keep observed vs imputed flags; score only on observed data
- Match the benchmark to the use case; always include persistence and interpolation
- Sensor swaps break year-over-year comparisons; version the sensor source
- 74,020 physically impossible readings quarantined (mostly turbidity)

**Visuals:** [findings summary](../multinode_visuals/07_findings_summary.png) as the base; optional [node reliability tiers](../multinode_visuals/05_node_reliability_tiers.png) for the analytics column. Note: its "Canal-mouth stress ~2 months before open bay" line is a 2021-22 hypothesis; omit or label it.

---

## 10. Can we forecast? Not yet, and here is the plan

**On slide**
- Hold-out test: persistence or a 7-day mean beats ridge almost everywhere (L1 Sep 2026, 1 day: persistence 0.38 h/day error, ridge 0.53; 7 days: 4.39 vs 6.36)
- Why: ~18 months of sensors, one autumn per node, 2 events in 60 months, none overlapping the sensors
- Next deck: forecast head on the GNN + FCM-derived features + external drivers (rain, tide, wind), judged against persistence on a held-out autumn

**Say**
- We now know the bar. Deck 2 tests whether combining the models clears it, with success criteria fixed before we run anything.

**Visuals:** left [NEW] horizon vs MAE chart from `analysis/north_bay_holdout_results.csv` (persistence, 7-day mean, ridge); right [NEW] roadmap: GNN + FCM + drivers -> forecast -> persistence check. Backup: [limitations](../multinode_visuals/17_limitations_and_next.png).

---

## Figures to build for this draft
- Slide 10: horizon vs MAE chart and roadmap (both new)
- Slide 2: relabel crest5 on the coverage heatmap (cosmetic)

## Keep out of the deck
- "The FCM predicts die-offs"
- "Sep 2026 is unprecedented"
- Spatial attention as bay transport (edge importance ~1e-4)
- FCM weights as causal effects
- 2025 vs 2026 claims at L0, L2, L7, p2, crest5 without the sensor-change caveat
