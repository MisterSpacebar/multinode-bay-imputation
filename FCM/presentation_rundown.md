# Biscayne Bay FCM: 10-slide presentation rundown

## Recommended framing

**Title:** From incomplete water-quality records to an interpretable network of relationships.

**Central message:** The project connects sensor reconstruction, historical field sampling, and a learned fuzzy cognitive map (FCM). The map is useful for organizing hypotheses and monitoring questions, but the current saved validation results do not demonstrate predictive skill or identify causal effects.

Allow roughly 12-15 minutes, plus questions. Each slide below separates audience-facing bullets from speaker notes. Values are drawn from the current saved analysis tables, not older narrative reports. Graphics are not a PowerPoint deck; they are presentation-ready PNG and SVG assets.

### Important evidence boundaries

- The daily physical FCM has **11 concepts**; the monthly nutrient FCM has **16**.
- Three concepts are exogenous: net water (rain minus potential evapotranspiration), minimum air temperature, and maximum air temperature. Net water is a weather-derived proxy, not measured canal discharge, runoff, or freshwater delivery.
- Learned signed coefficients are **lagged conditional associations**, not experimentally established causal arrows. Positive and negative do not mean beneficial and harmful.
- The saved monthly hindcast (corrected: it previously scored globally rescaled weights, which inflated predictions about 2.6x and produced negative R-squared) has **10 modelled concepts with in-sample R-squared 0.17 to 0.84**. Depth, pressure and turbidity are not modelled. Most of that fit is persistence: held-out, the FCM loses to "same as last month" on all 5 concepts with test data ([graphics/07](graphics/07_fcm_vs_persistence.png)).
- The validation is in-sample after refitting. Its "observation" mask is made after the loaders interpolate / fill data, and continuous data are already GNN-imputed. Do not describe all evaluation targets as independently observed.
- The fitting, simulation, and validation equations are not identical. Their discrepancies are limitations, not changes made by this presentation work.
- The CSV matrices do not retain the chosen lag, training dates, scaling parameters, or uncertainty. The date windows below describe the source code's intended datasets, not metadata certified by the matrices themselves.

## Slide 1 - Why an FCM after imputation?

**Takeaway:** Filling gaps and explaining relationships are different tasks.

**On-slide bullets**

- Biscayne Bay monitoring has incomplete and uneven sensor coverage.
- The graph neural network reconstructs missing sensor values.
- The FCM organizes relationships among weather, physical water quality, and nutrient / ecological indicators.
- Goal today: explore interpretable hypotheses and assess the evidence honestly.

**Graphic:** [Data-to-FCM pipeline](graphics/01_data_to_fcm.png). Show the left side first, then introduce the two FCM branches.

**Speaker notes:** Describe the GNN as the reconstruction layer and the FCM as the relationship layer. The tagged [model implementation](../model.py) is the spatial-attention / GRU imputation model, not the FCM itself. The FCM is implemented in [fcm.py](../fcm.py). Its concepts are variables, not monitoring stations. Regional aggregation means these maps are not geographical maps and do not preserve station-specific mechanisms.

**Suggested transition:** "Once we have a more complete time series, what relationships does it suggest, and how much should we trust them?"

## Slide 2 - Two data products, two temporal scales

**Takeaway:** The physical and nutrient maps answer related but nonidentical questions.

| | Physical FCM | Nutrient FCM |
|---|---|---|
| Resolution | Daily regional averages | Monthly merged regional series |
| Intended source window | March 2025-September 2026 | September 2021-September 2026 |
| Concepts | 3 forcing + 8 sensor variables | Same 11 + 5 nutrient / ecology variables |
| Main sources | GNN-imputed continuous sensors + weather | Historical surface grabs + monthly continuous data + weather |
| Added variables | None | pH, chlorophyll-a, Secchi depth, nitrite+nitrate, DIN |

**Graphic:** [Pipeline](graphics/01_data_to_fcm.png), emphasizing daily versus monthly branches.

**Speaker notes:** Physical sensor values are spatially averaged with weights of 1 for observed and 0.5 for imputed values. This is a heuristic weighting scheme, not an uncertainty model. Daily data are interpolated and forward/backward filled. The monthly loader combines historical Biscayne Bay surface samples and later continuous-sensor aggregates, applies limited interpolation and filling, and leaves many later nutrient values unavailable. Historical weather may use climatological filling outside available measurements. Shared variables, different sites, instruments, sampling frequency, and periods introduce a domain shift.

**Do not say:** "We measured all 16 variables continuously for five years."

## Slide 3 - How to read and construct the FCM

**Takeaway:** Nodes represent concepts; signed directed arrows represent learned lagged relationships.

**On-slide bullets**

- Each concept is min-max scaled to a normalized activation.
- Ridge regression predicts a target at a selected lag from the concept vector at time t.
- Source i -> target j is stored as W[i,j].
- Exogenous forcing receives no incoming edges.
- Weight magnitude indicates relative fitted strength, not a probability or physical-unit effect.

**Graphic:** Crop the air-temperature -> water-temperature -> dissolved-oxygen portion of the [ecology focus map](graphics/04_ecology_focus_map.png). Keep the sign legend.

**Speaker notes / equation panel**

```text
Fitting:       A_j(t + lag) approximately equals sum_i W_raw[i,j] A_i(t)
Weight export: W = W_raw / max(abs(W_raw))
Simulation:    A_free(t + 1) = sigmoid(A(t) W)
               A_forcing = fixed scenario values
Hindcast:      prediction(t + 1) = clip(A(t) W, 0, 1)
```

The Ridge fit has no intercept. Candidate lags are evaluated using a chronological tail, then weights are refit. Scaling and missing-input means currently use the full series, so lag-selection scores are not a leakage-free benchmark. The model retains self-connections: these can represent persistence and are hidden only in the maps. In the saved physical matrix, the oxygen self-weight is 1.0000 and water-temperature self-weight is 0.9387; do not present cross-concept ranking as the whole dynamical model.

**Teaching example:** The monthly water-temperature -> oxygen weight is -0.2216. This means a negative fitted lagged association conditional on the other inputs, not a causal oxygen-loss coefficient in mg/L per degree.

## Slide 4 - Physical map: recognizable structure and redundant measurements

**Takeaway:** Several of the strongest relationships connect physically related or overlapping measurements.

**On-slide bullets**

- Air temperature -> water temperature: positive fitted associations.
- Salinity <-> specific conductivity: a strong related-measurement pair.
- Depth <-> pressure: another physically coupled pair.
- Oxygen concentration <-> oxygen saturation: related, but distinct measurements.

**Graphic:** [Physical concept map](graphics/02_physical_concept_map.png). Use the [physical heatmap](../visualizations/08_fcm_heatmap.png) only as an appendix.

**Numbers for speaker notes**

| Directed relationship | Saved physical weight |
|---|---:|
| Minimum air temperature -> water temperature | +0.2985 |
| Maximum air temperature -> water temperature | +0.2781 |
| Salinity -> specific conductivity | +0.5283 |
| Specific conductivity -> salinity | +0.4595 |
| Depth -> pressure | +0.6932 |
| Pressure -> depth | +0.6624 |
| Oxygen mg/L -> oxygen saturation | +0.5626 |
| Oxygen saturation -> oxygen mg/L | +0.3833 |

**Speaker notes:** These are useful consistency checks, not discoveries of eight independent mechanisms. Some concepts contain overlapping measurement information, creating multicollinearity. This can inflate apparent connectivity or redistribute Ridge coefficients among related predictors.

**Display note:** The map shows 12 of 79 nonzero cross-concept edges: positive weights at W >= +0.25 plus negative weights at W <= -0.10 (a deliberately lower cutoff so negative examples stay visible). Four of the twelve are negative; none is larger in magnitude than -0.19. Isolated nodes mean "no displayed edge passes this filter," not "irrelevant."

## Slide 5 - Nutrient map: candidate environmental pathways

**Takeaway:** Adding nutrients expands the hypothesis space, but some fitted signs need scrutiny.

**On-slide bullets**

- Net water -> nitrite+nitrate and chlorophyll-a: positive associations.
- Warmer water -> lower oxygen: negative association.
- Nitrite+nitrate -> chlorophyll-a: positive association.
- DIN -> lower Secchi depth: negative association.
- Some signs conflict with simple ecological expectations.

**Graphic:** [Ecology-focused concept map](graphics/04_ecology_focus_map.png). Use the [16-concept overview](graphics/03_nutrient_concept_map.png) as backup.

| Directed relationship | Saved monthly weight | Presentation interpretation |
|---|---:|---|
| Net water -> nitrite+nitrate | +0.4238 | Candidate weather / nutrient linkage |
| Net water -> chlorophyll-a | +0.4301 | Candidate hydrologic / productivity linkage |
| Water temperature -> oxygen mg/L | -0.2216 | Compatible with an oxygen-stress hypothesis |
| Water temperature -> DIN | +0.4839 | May mix seasonality, loading and sampling effects |
| Nitrite+nitrate -> chlorophyll-a | +0.4003 | Candidate nutrient / productivity linkage |
| DIN -> Secchi depth | -0.3177 | Candidate reduced-clarity association |
| Chlorophyll-a -> Secchi depth | +0.4649 | Counterintuitive; investigate rather than endorse |

**Speaker notes:** Higher chlorophyll often suggests poorer clarity, so the positive chlorophyll -> Secchi coefficient is a diagnostic question. Likewise, the learned monthly DIN -> oxygen coefficient is positive (+0.2078), not the simple nutrient-loading -> oxygen-depletion narrative. Potential explanations include seasonality, collinearity, lag, aggregation, mixing of sampling regimes, or model misspecification. These explanations have not been isolated by the current analysis.

DIN includes nitrite+nitrate plus ammonium, so DIN and nitrite+nitrate are not independent chemical quantities. Avoid counting their connectivity as separate proof of a mechanism.

**Display note:** This is a disclosed, curated 10-edge subset across 8 of 16 concepts; not a reduced fitted model. The full overview shows 22 of 150 nonzero cross-concept edges: positive weights at W >= +0.42 plus negative weights at W <= -0.22 (five negative edges, lowest -0.34). The +0.42 cutoff is deliberately just low enough to keep the net water -> nitrite+nitrate (+0.42) and net water -> chlorophyll-a (+0.43) links visible.

## Slide 6 - Scenario exploration, not operational prediction

**Takeaway:** Scenarios reveal how this model responds to assumptions; they do not establish real-world intervention effects.

**On-slide bullets**

- Baseline: forcing held at mean normalized values.
- Wet surplus: net water set to 0.9.
- Heat-wave scenario: maximum air temperature 0.9, net water 0.05.
- Cold-and-dry scenario: maximum air temperature 0.1, net water 0.05.
- Read trajectories as exploratory normalized activations.

**Graphics:** [Physical scenarios](../visualizations/10_fcm_scenarios.png) and [nutrient scenarios](../visualizations/14_fcm_nutrient_scenarios.png). Prefer one per audience; move the other to backup. Retain their "normalized activation" axes.

**Speaker notes:** 0.9 means near the upper end of the dataset's normalized range, not 90% rain probability or an absolute temperature. The heat-wave and cold/dry scenarios change both maximum air temperature and net water, so they are not isolated temperature interventions. Forcing is fixed while endogenous concepts evolve under sigmoid dynamics. Simulation steps are labeled days / months in existing graphics, but the chosen lag is not saved and is not propagated as physical time into the scenario plots. Treat the x-axis as iterative model steps until that alignment is resolved.

Do not quote simulated activation as a nutrient concentration, a hypoxia probability, or an expected management benefit. Do not claim scenario rankings are robust without refitting / uncertainty tests.

## Slide 7 - Event-month context: what the measurements actually show

**Takeaway:** The two saved die-off event months have below-baseline oxygen, but they do not have identical environmental signatures.

**On-slide bullets**

- September 2021: oxygen mg/L -0.956 pooled standard deviations; pH -2.018.
- October 2022: oxygen mg/L -1.154; oxygen saturation -1.165.
- Water temperature: +0.934 in September 2021, but -0.088 in October 2022.
- DIN: +0.330 and +0.466, respectively.
- These are descriptive event-month associations, not FCM forecasts.

**Graphic:** [Event context comparison](graphics/06_event_context.png). Optional appendix: [site-level spatial maps](../visualizations/17_dieoff_spatial_maps.png).

**Speaker notes:** The saved anomaly table uses a pooled multi-year surface-sample baseline. Its preprocessing includes bay and canal sites while excluding inlet, reef and outfall types; this is a different site population from the FCM's historical Biscayne Bay-only loader. The baseline is not seasonally or site matched. Regional means can hide localized oxygen minima, and a negative oxygen anomaly alone does not establish hypoxia.

Do not call chlorophyll a strong event-month bloom signal here: the saved chlorophyll anomalies are -0.341 and -0.023. The September 2021 backcast is seeded in the event month because earlier data are unavailable; it cannot demonstrate advance warning. The October 2022 case is seeded in August, but remains an in-sample retrospective exercise, not independent event validation.

**Transition:** "Environmental plausibility matters, but predictive reliability must be tested separately."

## Slide 8 - Validation: the main limitation

**Takeaway:** In-sample fit is moderate, but the FCM does not beat "same as last month".

**On-slide bullets**

- Corrected hindcast: 10 modelled concepts, in-sample R-squared 0.17 (pH) to 0.84 (salinity); oxygen mg/L 0.77.
- Persistence is nearly as good in-sample (oxygen RMSE 0.107 vs 0.124 scaled units) and better held-out.
- Held-out (last 17 months): FCM loses to persistence on all 5 concepts with test data (oxygen mg/L RMSE 0.160 vs 0.096).
- Nutrient concepts have no held-out data (grab samples stop earlier).
- The strongest claim today is interpretability and hypothesis generation.

**Graphic:** [FCM vs persistence](graphics/07_fcm_vs_persistence.png) (new), [validation scorecard](graphics/05_validation_scorecard.png). Optional backup: [hindcast scatter panels](../visualizations/21_fcm_hindcast_scatter.png).

**Speaker notes:** Earlier versions of this slide showed all 13 R-squared values negative. That was a scoring error: the hindcast used weights globally rescaled to [-1, 1] for display and sigmoid simulation, which inflated predictions about 2.6x and pushed them to the clip ceiling. The hindcast now uses the unscaled ridge weights (`learn_fcm_weights(..., normalise=False)`) and skips concepts the FCM never fitted.

The corrected scores are still in-sample, and the mask includes loader-filled / imputed values. Held-out numbers use only 17 monthly steps, during a period when oxygen trends downward, so they are noisy and shift-sensitive; treat them as "no evidence of skill" rather than a precise skill estimate. Simulation still applies a sigmoid while hindcasting clips a linear update. RMSE is in normalized activation space, not mg/L.

**Do not hide:** The correction, and that persistence is the bar the FCM has not cleared.

## Slide 9 - Who can use these findings?

**Takeaway:** The project has useful applications for both audiences, but at different evidence levels.

**Graphic / slide layout:** Two side-by-side text panels; place a small copy of the [ecology focus map](graphics/04_ecology_focus_map.png) between them only if labels remain readable.

### Environmental science panel

1. **Monitoring design:** Measure temperature, oxygen, clarity and nutrient species together around weather changes to test candidate pathways.
2. **Field hypotheses:** Test whether weather-associated nutrient and chlorophyll patterns persist at matched sites and seasons.
3. **Event heterogeneity:** Below-baseline oxygen occurs in both saved event months, but temperature and pH signatures differ; investigate multiple pathways rather than one universal trigger.
4. **Spatial follow-up:** Regional relationships motivate station-level and canal-vs-bay analyses, not site-specific attribution from the regional map.

### Data science / analytics panel

1. **Interpretable representation:** Signed variable networks offer a stakeholder-readable complement to GNN reconstruction.
2. **Provenance-aware pipelines:** Carry measured / reconstructed / interpolated / climatology-filled flags end-to-end.
3. **Redundancy and domain shift:** Related oxygen and conductivity measurements, plus field-vs-sensor regimes, complicate coefficient interpretation.
4. **Evaluation discipline:** A display-rescaled weight matrix silently turned a modest model into a "negative R-squared" one; against persistence and held-out data, plausible diagrams are still not performance evidence.

**Speaker notes:** Environmental users can use the map to prioritize measurements and questions today. Analytics users can use it to design better data contracts and evaluation. Neither audience should use it yet as a validated early-warning system or causal management optimizer.

## Slide 10 - What would make the FCM decision-ready?

**Takeaway:** Keep the communication layer, but strengthen the evidence layer.

**On-slide priorities**

1. Preserve original observation provenance and immutable input snapshots.
2. Save lag, date window, site selection, scaling, regularization and fitting metadata.
3. Align learning, hindcast and simulation mathematics.
4. Use train-only preprocessing and rolling chronological holdouts.
5. Compare with persistence, seasonal mean and a straightforward lagged Ridge / autoregressive baseline.
6. Estimate edge uncertainty; test signs across seasons, sites and aggregation choices.
7. Validate any event-warning thresholds against independent events and measured oxygen.

**Graphic:** [Validation scorecard](graphics/05_validation_scorecard.png) reduced to the left half; put the priorities on the right. Alternatively reuse the [pipeline](graphics/01_data_to_fcm.png) with provenance and holdout annotations in your slide editor.

**Speaker notes:** First evaluate the unrescaled Ridge predictions at the actual fitted lag to distinguish regression quality from FCM transformation effects. Then choose and validate a coherent bounded update rule. Treat expert ecological constraints as explicit assumptions rather than silently replacing inconvenient learned signs. For uncertainty, use time-aware resampling rather than assuming independent monthly samples.

**Closing line:** "The FCM is currently a transparent hypothesis map; the next step is to make its relationships stable and its predictions independently testable."

## Addendum - the north-bay story with data through September 2026

Use these as slides 11-13 (or an appendix). All numbers are observed-only (imputed values excluded); see `analysis/north_bay_hypoxia_monthly.csv`.

**Slide 11 - Where the sensors are.** Graphic: [coverage heatmap](../multinode_visuals/02_coverage_heatmap.png). Message: the north bay only has continuous sensors from March 2025; 2021-2024 grab sites stop at about 25.83 N. Since June 2026 several nodes also take buoy data (L0=Haulover, L2=CREST3, L7=CREST4, p2=Royal Castle, crest5=CREST5+Kampong), so only L1 and L6 are like-for-like across the whole record.

**Slide 12 - Low oxygen is a recurring late-summer/autumn season.** Graphics: [hypoxia heatmap](../visualizations/30_north_bay_hypoxia_heatmap.png) and [2025 vs 2026](../visualizations/31_north_bay_hypoxia_2025_vs_2026.png). L1 canal spent 22.2 h/day below 4 mg/L in Sep 2025 and 23.7 in Sep 2026; L2 stayed at 20-23 h/day through Dec 2025. Sep 2026 is not unique, it repeats Sep 2025. The south reference sites (L7, crest5) are equally low in 2026, so this is not only a north-bay canal effect. L0 (Haulover Inlet, ocean-flushed) stays near 0 h/day in 2026, which is consistent with its change to the inlet buoy.

**Slide 13 - What we can and cannot forecast.** Graphic: [validation scorecard](graphics/05_validation_scorecard.png). On a September hold-out, persistence or a 7-day trailing mean beats a ridge model almost everywhere (`analysis/north_bay_holdout_results.csv`). The FCM does not beat persistence held-out on the 5 concepts with test data (graphics/07). Message: with about 18 months of continuous data and one autumn season per sensor, the evidence supports description and monitoring design, not forecasting or die-off prediction.

## Making the maps readable

The new graphics use a conventional directed concept-map presentation: distributed labeled boxes, signed arrows, curved reciprocal links and semantic placement. An FCM does not require a unique standard layout; clarity matters more than circular, column or force-directed geometry.

- **Main-slide map:** Use the 8-concept, 10-edge [ecology focus map](graphics/04_ecology_focus_map.png), not all 150 nonzero monthly cross-concept edges.
- **Overview / appendix:** Use the [16-concept strong-edge map](graphics/03_nutrient_concept_map.png) or the [full heatmap](../visualizations/12_fcm_nutrient_heatmap.png).
- **Put labels inside equal-sized boxes.** This avoids external labels colliding with edges and avoids implying that node size is a validated importance score.
- **Encode sign twice:** teal solid for positive, orange dashed for negative. This remains distinguishable in grayscale and avoids relying on red/green alone.
- **Label the filter and omissions.** Filter thresholds are display choices, not significance tests. Isolated nodes are not necessarily irrelevant. Self-loops remain in the model.
- **Keep clear arrowheads and signed weights.** Reciprocal relationships are separate arrows, not a single ambiguous bidirectional line.
- **Use 16:9 and SVG.** All new figures have 2880 x 1620 PNG exports and matching SVG exports. The generator keeps SVG text as text for editing.
- **Progressively reveal pathways.** Discuss weather -> temperature -> oxygen, then weather -> nutrients -> chlorophyll / clarity. Do not add unlearned expert arrows without a distinct visual style and disclosure.
- **Do not overinterpret filtered maps.** The strong-edge overviews use a lower display cutoff for negative weights than for positive ones so that negative examples are visible; negative links are generally weaker, so do not compare line widths across signs as if the cutoffs were equal.

## Files, regeneration and checks

Run from the repository root:

```powershell
& '.\.venv\Scripts\python.exe' fcm_presentation.py
& '.\.venv\Scripts\python.exe' -m unittest test_fcm_presentation
```

[The generator](../fcm_presentation.py) reads the saved physical / nutrient weight matrices, hindcast metrics and event anomalies. It does not train models, fetch data, edit saved analyses, or overwrite the original visualizations. PNG and SVG outputs are stored in [graphics](graphics/).

The existing scenario figures are referenced, not recomputed here. If those figures and the CSVs were produced by different runs, regenerate / version the original analysis together before final presentation.

### Evidence sources

- [FCM implementation](../fcm.py): loading, normalization, Ridge fitting, lag selection and sigmoid simulation.
- [Validation implementation](../fcm_validate.py): in-sample hindcast, observation-mask construction and event seed dates.
- [Physical weights](../analysis/fcm_weights_physical.csv).
- [Nutrient weights](../analysis/fcm_weights_nutrient.csv).
- [Hindcast metrics](../analysis/fcm_hindcast_metrics.csv).
- [Event anomalies](../analysis/dieoff_anomalies.csv).
- [Event-analysis implementation](../dieoff_analysis.py): pooled baseline and site population.

Older figure reports use some stronger causal / predictive wording and can predate current outputs. Use the current tables and caveats above rather than copying those claims.
