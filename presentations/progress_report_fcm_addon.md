# Progress report add-on: FCM (6 slides)

Goes after the progress report (GNN imputation status). Compact bullets; visuals are existing files.

**A1. What is an FCM?**
- Concept map: boxes = variables, signed arrows = lagged links, learned by ridge regression (1 step = 1 day or 1 month)
- Teal solid = positive, orange dashed = negative; weights are associations, not causes
- Gives an interpretable "what moves what" view next to the GNN's reconstruction
- Visual: [data to FCM pipeline](../FCM/graphics/01_data_to_fcm.png), [physical map](../FCM/graphics/02_physical_concept_map.png)

**A2. What the FCM found (bay-wide, 2021-26)**
- Air temp -> water temp (+0.52); net water (rain - PET) -> nitrate and chlorophyll (+0.42, +0.43)
- Warmer water -> lower DO (-0.22); chlorophyll -> clearer/less clear water (Secchi +0.46); DIN -> Secchi (-0.32)
- Useful for choosing what to measure; not a validated mechanism
- Visual: [ecology focus map](../FCM/graphics/04_ecology_focus_map.png)

**A3. North-bay check on 2025-26 sensor data**
- Observed days only, nodes L1, L2, L6, p2 bay
- Salinity is the only clear DO driver (+0.08 mg/L per ppt same day, interval excludes 0); temperature effect is small
- Yesterday's value dominates (persistence 0.91-0.95)
- Visual: [north-bay FCM map](../visualizations/32_fcm_north_bay_2025_26.png)

**A4. How well does the FCM predict? Not beyond persistence**
- In-sample monthly R^2 0.17-0.84, but mostly "this month predicts next"
- Held-out (last 17 months): FCM loses to "same as last month" on 5 of 5 testable concepts (DO RMSE 0.160 vs 0.096)
- Event backcasts (2021, 2022) flatten to a fixed point and miss the DO dip
- Correction: an earlier scorecard showed all-negative R^2 from a weight-rescaling error; fixed (see speaker note)
- Visual: [FCM vs persistence](../FCM/graphics/07_fcm_vs_persistence.png) (main), [Oct 2022 backcast](../visualizations/24_fcm_backcast_oct2022.png) (backup)

**A5. FCM vs GNN on the same question (DO, north bay)**
- FCM: ties persistence one day ahead (R^2 0.838 vs 0.833)
- GNN (DO masked, Jun-Sep 2026): beats baselines at L1 and p2 bay, loses at L2 and crest5, misses nightly drops toward 1 mg/L
- Both rank salinity above temperature; they disagree on size
- Caveat: test window overlaps the June 2026 sensor change; nowcast, not forecast
- Visual: [FCM vs GNN](../visualizations/33_fcm_vs_gnn_north_bay.png)

**A6. For discussion**
- Target: hours/day below 4 mg/L, 1-14 days ahead, scored against persistence
- Role of the FCM: structural prior (drivers, signs, lags), not the forecaster
- Next model: tabular + external drivers first, then a GNN forecast head
- Data asks: more north-bay months, nearby agency DO history, rain, tide, wind, water-management data
- Open question: how to handle the Jun 2026 sensor change in training and testing

## Speaker note for A4
The earlier scorecard (all 13 R^2 negative, DO RMSE 0.37 vs persistence 0.12) scored weights that had been rescaled to [-1, 1] for display (about 2.6x too large). Scored with the unscaled ridge weights, the FCM is a decent in-sample one-step fit but still no better than persistence held-out. Held-out n is only 17 months during a downward DO trend, so read it as "no evidence of skill", not a precise skill estimate.
