# Physical FCM v2: station-aware, observed-only lagged concept maps

**Status:** experimental, retrospective. Nothing here establishes causality
or operational hypoxia alerts. This new package leaves root `fcm.py`,
`fcm_validate.py`, `fcm_predictive_check.py`, historical nutrient maps,
figures, and analysis files unchanged.

## Purpose and differences from legacy FCM

The old FCM reads seven merged-node exports, averages sensor readings across
very different environments, interpolates missing values from the future,
normalizes on the complete series, rescales ridge coefficients, and then
executes a sigmoid function that was *not* the fitted predictive function.

This package:

- Reads only `trustworthy_outputs/daily_targets.csv` created from
  **observed** sensor values (the GNN-imputed measurements are never target
  labels or model predictors).
- Preserves each node. **No spatial averaging or imputation.**
- Uses local-calendar day grid with actual missing dates represented by NaN.
  Hypoxia hours are an observed lower bound when the day's coverage is <100%.
- Fits every model using only concept pairs with their **future target dates
  strictly before the forecast issue date**. Centering, scaling, coefficient
  learning and output transforms are train-only.
- Predicts one *specific future day* at each of 1, 3 or 7 days ahead.
  A 7-day forecast is NOT the sum or average of seven intervening days.
- Provides **two concept sets**:
    - DO: `hours_lt4`, `odo_mean`
    - Physical: `hours_lt4`, `odo_mean`, `temp_mean`, `sal_mean`
- Fits two activation functions:
    - `linear`: regularized standardized transition (lagged linear map)
    - `sigmoid`: regularized logit-transformed bounded transition (nonlinear
      FCM-style concept activation). Fitted coefficients are used directly
      under exactly the chosen activation; no postfit max-weight rescaling.
- Fits each target concept separately and records missing target-pair counts.
  DO/hypoxia output always requires `--min-train` valid examples.
- Writes signed concept associations, and optional hypothetical +1 degree C
  and +1 ppt input sensitivity for the **physical** map. These are **not
  causal interventions**, nutrient effects, or real weather forecasts.
- Tests every forecast against persistence on **that exact issue date,
  target date, node and horizon**. All-model common-case rankings are
  additionally reported when trustworthy baseline predictions exist.

## How to run

From the repository root, after merging:

```bash
python -m unittest discover -s fcm_v2/tests -v
python -m trustworthy.quality
python -m trustworthy.coverage_report
python -m trustworthy.backtest
python -m fcm_v2.evaluate
```

The new FCM does not need another GNN training run if the current
`trustworthy_outputs/daily_targets.csv` is already regenerated.

For a quick comparison on the original months, or to limit compute:

```bash
python -m fcm_v2.evaluate --months 2025-09,2026-09 --activation linear
```

Default run iterates over all recorded months and both activations for all
three horizons, refitting on historical cases at each issue date. This is a
slower but strictly causal-time retrospective backtest.

Use `--no-edges` to skip rolling signed edge and scenario-sensitivity
outputs (model predictions and metrics are unchanged).

## Files written to fcm_v2_outputs/ (ignored by Git)

- `predictions.csv`: raw forecast outcomes, with paired persistence values.
- `summary.csv`: per node/month/horizon/FCM variant RMSE, MAE, hypoxic-day
  absolute error, sample counts and **paired RMSE skill**.
- `exclusions.csv`: explicit model eligibility reasons and counts.
- `learned_edges.csv`: signed standardized linear or logit transition
  weights (varying by training end date, node and horizon).
- `scenario_sensitivity.csv`: changes in model predictions when an issue-day
  sensor input is perturbed by +1 C or +1 ppt, never labeled causal.
- `all_models_common_cases.csv`: trustworthy baseline and FCM models
  evaluated on their **exact shared** issue/target dates. Missing required
  features can reduce this overlap dramatically.

## Interpreting results

A model winning in a percentage of station-month groups is not the same
as winning overall. Examine paired error magnitudes, counts, hypoxic target
days, station-month variability, and persistence-relative RMSE skill.

A high positive autoregressive DO self-weight means the FCM may be little
more than persistence with extra parameters. Hypoxia hours are bounded to
[0, 24], and some sites may have almost no variation or low-DO events;
models may therefore produce limited or misleading edge weights. A map is
not promoted to an early warning system without independent event-level
testing and uncertainty calibration.

The sigmoid targets use a fixed, declared bounded logit scale, clipped by
EPS around endpoints (0h and 24h). That is a practical regularized
approximation, not a globally optimized neural sigmoid FCM. Other
continuous targets use conservative predefined physical sanity bounds.
All signs/strengths are **conditional lagged associations**; collinearity,
seasonality, and measurement errors can change them. The linear model's
coefficients are directly learned and used; the sigmoid model's coefficients
are logit-scale and are *not* numerically comparable to linear coefficients.

The current trustworthy targets use complete day-of-issue information,
so forecasts are issued **at the end of the day**. The data may include
unknown sensor replacements and unreviewed flatlines (flagged separately).
No future rainfall, discharge, tides, weather forecasts or nutrients are used.

## Next integration experiments (not in this PR)

1. Compare against the trustworthy DO-only and DO+physical Ridge benchmarks
   on identical cases. Require measurable improvements in RMSE **and** errors
   on hypoxic target days before expanding.
2. Incorporate weather/tide/flow covariates **only** once their availability
   timing is documented. Avoid retrospective climate interpolation.
3. Add predeclared ecological links (respiration, stratification, flushing)
   as hypotheses or regularization constraints, distinguishable from fitted
   coefficients. Test their predictive impact via ablation.
4. Keep the sparse 2021–2024 survey nutrient FCM separate by station/depth
   and survey date. Do not manufacture monthly nutrient observations after
   surveys stop.
5. Explore GNN spatial embeddings only after reliable feature coverage and
   common-case FCM results justify the extra modeling complexity.

## Post-fit diagnostics: model uncertainty and edge reliability

Once `python -m fcm_v2.evaluate` and `python -m trustworthy.backtest`
have generated their own ignored local CSVs, run:

```bash
python -m fcm_v2.diagnostics
```

This reads `fcm_v2_outputs/learned_edges.csv`,
`fcm_v2_outputs/predictions.csv`,
`trustworthy_outputs/backtest_predictions.csv`, and (when present)
`fcm_v2_outputs/scenario_sensitivity.csv`. It does not need any source
instrument files or `deployments.csv`. You can run
`--bootstrap 5000` for more Monte Carlo replicates.

Outputs in `fcm_v2_outputs/diagnostics/`:

- `paired_uncertainty.csv`: matched-node/issue/target comparisons of the
  physical FCM with persistence, the DO-only FCM and Ridge physical.
  Includes approximate 95% station-stratified station-month resampling
  intervals for RMSE difference, sample sizes and hypoxic-day errors.
- `edge_stability.csv`: signed coefficient medians, 10–90% quantiles,
  origin-month counts, meaningful sign flips, near-zero fraction, early/late
  medians and exploratory stability flags. Consecutive refitted models
  overlap strongly; stability **is not** independent scientific validation.
- `edge_do_summary.csv`: count of stations with positive, negative and
  exploratory-stable incoming DO/hypoxia edges by horizon.
- `leave_one_station_out.csv`: changes in the candidate-vs-reference
  result after omitting each station, detecting dominance by one location.
- `sensitivity_audit.csv`: flags unusually large fitted responses to +1 C
  or +1 ppt issue-day perturbations. Large effects can arise from
  collinearity, narrow station-specific predictor ranges, sparse data or
  extrapolation; they are *not* causal effects.

As an external check, the uploaded November 2026 v2 files had seven-day
common-case RMSE about 7.07 h (linear physical), 7.49 h (linear DO),
6.96 h (Ridge physical) and 7.94 h (persistence) across 473 forecasts
and six stations. Exploratory station-month resampling gave FCM minus
persistence approximately -0.87 h (95% interval -1.50 to -0.17 h),
FCM minus DO-only approximately -0.42 h (-0.85 to +0.06 h), and FCM
minus Ridge approximately +0.10 h (-0.01 to +0.21 h). Intervals are
approximate, do not account for all seasonal dependence or post-hoc
model selection, and should not be presented as confirmatory tests.
The +1 ppt salinity sensitivity was near +10 h at two stations:
review the range, collinearity and scientific plausibility before
interpreting these coefficients.

**Critical interpretation**: the current direct-horizon FCM with a
linear activation is a vector of fitted Ridge regressions, with no
recursive concept feedback in the target forecast. It does not add
FCM-specific predictive structure relative to an identical Ridge
regression with the *same features and penalty*. The existing
`ridge_physical` baseline has a different feature set, so an
FCM-vs-Ridge comparison is not itself a controlled architectural ablation.
A true recurrent FCM should be evaluated only after it is specified,
fitted and tested under the same leakage-free procedure.
