# Deck 2: Forecasting low oxygen with GNN + FCM + more

Placeholder: nothing here has been run. `[RESULT]` = to fill in after experiments. Follows [Deck 1](deck1_gnn_to_fcm_scaffold.md).
Question: can we forecast hours/day below 4 mg/L, 1-14 days ahead, better than persistence in the north bay?

**1. Recap: the bar to clear**
- Persistence beats FCM (held-out, 5 of 5) and ridge (Deck 1, slides 14 and 20)

**2. Target**
- Hours/day below 4 mg/L (and 2); optional "hypoxic day" flag
- Horizons 1, 3, 7, 14 days
- Visual: [NEW] target definition

**3. Evaluation protocol**
- Rolling-origin splits, train-only scaling, observed-only scoring
- Baselines: persistence, 7-day mean, seasonal mean, lagged ridge/AR
- Visual: [NEW] split schematic

**4. Role of each model**
- GNN: spatial state estimator + forecast head ([model.py](../model.py)); risk: edges barely matter, trained for imputation
- FCM: structural prior (drivers, lags, signs); risk: no better than persistence, bay-wide averages
- Something else: gradient-boosted/ridge on lags + external drivers (rain, tide, wind, water-management ops, air temp); risk: few autumns, alignment
- Visual: [NEW] component diagram

**5. Candidate A: GNN forecast head** `[RESULT]`
- Horizon targets replace mask-and-reconstruct

**6. Candidate B: FCM-informed** `[RESULT]`
- Lagged concept states as inputs or sign constraints; ablation vs without

**7. Candidate C: external drivers + tabular** `[RESULT]`
- Run first; if it can't beat persistence, nothing fancier will

**8. Hybrid / ensemble** `[RESULT]`
- Keep only parts with measurable skill

**9. Probabilistic output** `[RESULT]`
- Probability of a hypoxic day

**10. Skill by horizon and node** `[RESULT]`
- Skill score vs persistence

**11. 2021-22 back-test** `[RESULT]`
- Monthly grab data only; cannot test daily hours

**12. Decision and data asks**
- More north-bay months; nearby agency DO history; rain, tide, wind, water-management data

## Experiment order
1. Strong baselines, rolling origin
2. Tabular + external drivers
3. GNN head, with/without FCM inputs (GNN, GNN+FCM, tabular, tabular+FCM)
4. Ensemble + probabilistic output
5. Event back-tests

## Success criteria (set before running)
- Skill > 0 vs best of persistence/7-day mean at >=3 horizons on >=2 north-bay nodes, on a fully held-out autumn
- Observed-only scoring, no scaling leakage
- Holds after excluding nodes whose sensors changed in Jun 2026, or flag as source-dependent
