# Deck 1: From GNN imputation to the FCM

Arc: fix the data -> map what moves together -> check what the newest data show -> can we forecast? (leads into [Deck 2](deck2_forecasting_scaffold.md))
Data: sensors Mar 2025-Sep 2026; monthly grab samples Sep 2021-Sep 2026. FCM speaker notes: [../FCM/presentation_rundown.md](../FCM/presentation_rundown.md)

## Part A: GNN

**1. Title / question**
- Can a patchy sensor network tell us about hypoxia in north Biscayne Bay?
- Visual: [bay map](../report/canal_stress_map_v2.png)

**2. The data problem**
- Observed coverage 0-73% by node and feature
- Most sensor-time is missing or imputed
- Visual: [coverage heatmap](../multinode_visuals/02_coverage_heatmap.png)

**3. GNN in one slide**
- Stations = nodes, distance = edges, attention = who informs whom
- Visual: [GNN concept](../multinode_visuals/13_gnn_concept.png)

**4. Architecture and training**
- Encoder -> 3 attention layers -> GRU -> decoder
- Mask-and-reconstruct, chronological split
- Visual: [architecture](../multinode_visuals/01_architecture_diagram.png), [training pipeline](../multinode_visuals/08_training_pipeline.png)

**5. What it does to a gap**
- Raw vs imputed, long and short gaps
- Visual: [long gap](../multinode_visuals/10_before_after_imputation.png), [short gap](../multinode_visuals/11_before_after_small_gap.png)

**6. What it learned**
- Salinity and temperature carry the signal
- Spatial edges add little (importance ~1e-4)
- L7-crest5 attend only to each other: fragile pair
- Visual: [channel importance](../multinode_visuals/12_channel_importance_clean.png), [attention network](../multinode_visuals/03_attention_network_annotated.png)

**7. Honest scorecard**
- Beats climatology and spatial mean
- Loses to linear interpolation on random gaps (DO 0.52 vs 0.48, temp 1.95 vs 1.41)
- Value is structural gaps, where interpolation has nothing to use
- Visual: [baseline overview](../multinode_visuals/15_baseline_overview.png)
- Bridge: gap-free record, but no explanation of drivers -> need an interpretable map

## Part B: FCM

**8. Why an FCM after imputation**
- GNN reconstructs; FCM shows lagged associations transparently
- Visual: [data to FCM](../FCM/graphics/01_data_to_fcm.png)

**9. Two products**
- Daily physical FCM: 11 concepts
- Monthly nutrient FCM: 16 concepts, grab data to 2024 only

**10. How to read it**
- Signed, weighted, lagged arrows
- Weights are ridge associations, not causal effects
- Visual: [physical map](../FCM/graphics/02_physical_concept_map.png)

**11. Ecology pathways**
- Weather -> temperature -> oxygen
- Weather -> nutrients -> chlorophyll / clarity
- Visual: [focus map](../FCM/graphics/04_ecology_focus_map.png); backup: [full nutrient map](../FCM/graphics/03_nutrient_concept_map.png)

**12. Scenarios**
- "What if" runs for hypotheses only
- Visual: [scenarios](../visualizations/10_fcm_scenarios.png)

**13. 2021 vs 2022 events**
- 2021: freshwater-dilution signature
- 2022: saline, nutrient-loading signature
- Visual: [event context](../FCM/graphics/06_event_context.png), [die-off overview](../multinode_visuals/14_dieoff_overview.png)

**14. Validation: main limitation**
- Monthly hindcast R^2 0.17-0.84 in-sample (mostly persistence)
- Held-out: persistence beats FCM on 5 of 5 testable concepts
- Backcasts collapse to a flat fixed point
- Visual: [FCM vs persistence](../FCM/graphics/07_fcm_vs_persistence.png), [scorecard](../FCM/graphics/05_validation_scorecard.png)
- Bridge: useful for reasoning, not predicting -> what do the newest data say?

## Part C: Findings through Sep 2026

**15. Where the sensors are**
- North bay continuous only since Mar 2025
- Grab sites stop ~25.83 N; no overlap with 2021-22 events
- From Jun 2026 L0, L2, L7, p2, crest5 also use buoy data; only L1 and L6 are like-for-like
- Visual: coverage heatmap, or [NEW] source timeline

**16. Low oxygen is a recurring season**
- L1 canal: 22.2 h/day below 4 mg/L (Sep 2025), 23.7 (Sep 2026)
- L2: 20-23 h/day through Dec 2025
- Visual: [hypoxia heatmap](../visualizations/30_north_bay_hypoxia_heatmap.png)

**17. 2025 vs 2026**
- Same pattern both years
- South references (L7, crest5) equally low in 2026: not just a north-canal effect
- crest5 worst: 22-24 h/day <4, 10-13 h/day <2 (Jul-Sep 2026)
- Visual: [2025 vs 2026](../visualizations/31_north_bay_hypoxia_2025_vs_2026.png)

**18. For environmental scientists**
- Low-oxygen season timing; monitoring priorities
- Canal-mouth stress ~2 months before open bay (2021-22; hypothesis)
- Depth, pressure, spec. cond. are local, redundant network-wide
- Visual: [findings summary](../multinode_visuals/07_findings_summary.png)

**19. For data scientists**
- Keep observed vs imputed flags (provenance)
- Baselines are the only honest metric
- Sensor swaps break like-for-like comparisons
- 74,020 impossible readings quarantined (mostly turbidity)
- Visual: [node reliability tiers](../multinode_visuals/05_node_reliability_tiers.png)

## Part D: Can we forecast?

**20. Hold-out test**
- Persistence or 7-day mean beats ridge almost everywhere
- Only exception: p2 bay at 1 day (4.02 vs 4.19 h/day), not meaningful on 30 days
- Visual: [NEW] horizon vs MAE from `analysis/north_bay_holdout_results.csv`

**21. Why: evidence limits**
- ~18 months of sensors, one autumn per node
- 2 events in 60 months, none overlapping the sensors
- Sensors swapped Jun 2026
- Visual: [limitations](../multinode_visuals/17_limitations_and_next.png)

**22. The plan**
- GNN + FCM + something else, judged against persistence
- Visual: [NEW] roadmap

## Keep out of the deck
- "FCM predicts die-offs" (it does not beat persistence)
- "Sep 2026 is unprecedented" (Sep 2025 was equally bad)
- Spatial attention as bay transport
- 2025 vs 2026 claims at changed nodes without the Jun 2026 caveat
- FCM weights as causal

## Figures to build
- Horizon vs MAE chart (slide 20)
- Source timeline (slide 15)
- Roadmap (slide 22)
- Relabel `consolidated_crest5` on the coverage heatmap
