# Phase 4 — Observed-only environmental GAM pilot

The aim is to characterize nonlinear associations between local measured temperature/salinity and (a) daily minimum dissolved oxygen; (b) **next-day recovery during an already observed hypoxic episode**. These are descriptive/prognostic analyses, **not validated new-onset warnings or causal explanations**.

## Inputs and provenance

1. `trustworthy_outputs/daily_targets.csv` from `python -m trustworthy.quality` — observed-only, coverage-gated measured DO, temperature and salinity; NOT a table of GNN reconstructions.
2. `trustworthy_outputs/episode_dynamics/recovery_risk_days.csv` from `python -m trustworthy.episode_survival` — only identifiable onset-defined episode-days with an observed, eligible next-day outcome. Unknown future observations are not labeled recovery or persistence.

Require same-station, same-date matching and >=95% observed DO coverage. Temp/sal must be finite. The local-day one-day DO trend is computed only from two adjacent, eligible station-days; do not interpolate missing dates or copy measurements between stations. Zero, flatline and spike suspects remain in the primary fit with their QC flags. Repeat analyses excluding flagged cases as a sensitivity check, rather than silently filtering them.

**Important:** Daily `odo_min` is a *retrospective outcome*, not a prediction issue-time variable. The severity model uses same-day temp/sal and is explanatory rather than a weather-based ahead-of-time DO forecast. Recovery models use end-of-day measurements to associate conditions with *next-day* recovery while a hypoxic episode is underway. A new-onset warning system needs a separate risk set of currently non-hypoxic days and subsequent calibration/verification.

## Run

Install the GAM prerequisites:

```bash
python -m pip install 'statsmodels>=0.14,<0.16' 'scipy>=1.11,<2' 'numpy>=1.26,<3' 'pandas>=2.2,<3'
```

Generate the prior observed-only inputs, then run:

```bash
python -m trustworthy.environmental_gam \
  --daily trustworthy_outputs/daily_targets.csv \
  --risk trustworthy_outputs/episode_dynamics/recovery_risk_days.csv \
  --out-dir trustworthy_outputs/environmental_gam \
  --min-coverage 0.95 --min-train 100 --alpha 10
```

Use `--no-trend` to omit the previous-day DO trend from the recovery GAM when too many eligible risk-days lack consecutive-day history. This produces a **different matched evaluation population**; do not directly compare different runs' scores as though they used the same cases. Source files remain unchanged.

## Pre-specified models

- Severity baseline: station fixed effects + annual sine/cosine seasonality (Gaussian GLM).
- Severity GAM: same baseline plus penalized cubic B-splines for **same-day observed temperature and salinity**. Response = observed daily minimum DO mg/L. These curves are *associations*, not advance predictions.
- Recovery baseline: threshold-specific cloglog GLM with station fixed effects + annual harmonic terms + log elapsed episode days + same-day measured hours below threshold (6-hour units). Outcome = observed **next-day recovery**.
- Recovery GAM: adds penalized cubic temperature and salinity splines, plus optional lagged DO trend calculated from previous and current eligible days. No regional weather proxies added in this initial phase; these are future controlled sensitivity/ablation additions only.

Spline basis is fixed at 4 basis functions per predictor, degree 3, with a fixed smoothing penalty alpha=10 (rather than searching many hyperparameters on a small event sample). For test measurements outside the training range, the predictor is clipped to the training support; no unsupported spline extrapolation. Station effects require an already-seen station; previously unseen stations are excluded from that fold. The smooth-effect curves hold the station, date, episode stage and other measurements fixed at representative values, vary one environmental covariate across the observed 5th–95th percentile, and **are not causal effects or uncertainty intervals**.

## Honest time-block validation

The comparison uses expanding calendar-month evaluation windows. The model for each month trains on preceding months only; recovery training uses `next_date < first_test_issue_date`, so an incomplete outcome window never leaks into the training set. Paired baseline and GAM use the **same eligible station/date cases**, each with all measured covariates. The primary comparison metrics are mean absolute error for daily-minimum DO and Brier score for next-day recovery. The test is a diagnostic of temporal generalization, not an operationally calibrated warning system. If too little data or too few recovery classes exist, a month is explicitly marked as unavailable.

The joined sample can be much smaller than Phase 3's 662 risk-days because temperature/salinity and *consecutive eligible DO trend* must exist. Report the attrition, positives, number of eligible stations and test months before interpreting gains. Avoid strong results when events are sparse and station records span different years.

## Outputs

- `observed_recovery_features.csv` — linked, observed-only, complete-case recovery predictors and labels, including DO trend and suspect flags.
- `full_fit_summary.csv` — whole-sample descriptive baseline/GAM fits, AIC and effective degrees of freedom. Do **not** interpret these as held-out scores.
- `gam_smooth_effects.csv` — conditional environmental GAM curves for temp/sal, with held-constant reference values and sample sizes.
- `figures/` — separate exploratory smooth-effect plots for each response/threshold and predictor (only if matplotlib installed).
- `time_block_predictions.csv` — **out-of-sample** matched predictions from monthly expanding-origin folds.
- `paired_scores.csv` — matched-case baseline and GAM MAE/RMSE or Brier, and skill relative to the matched baseline.
- `fold_diagnostics.csv` — month-specific training/test sizes, positives and exclusions; no invented scores for unsupported folds.
- `environmental_gam_metadata.json` — exact SHA-256 of both inputs, coverage, penalty, history and sample counts.

## Stop/go criteria for warning indicators

1. Confirm the input hashes and the observed-only sample/population, and inspect model convergence and attrition.
2. Inspect whether nonlinear temp/sal patterns are stable across stations/time folds, especially with suspect QC exclusion.
3. Require a **positive and uncertainty-supported held-out Brier improvement** over the simple recovery model before claiming that environmental conditions add prognosis during active episodes.
4. For advance onset warnings, build a separate **issue-time-only** onset model for 1-day horizon using eligible currently non-hypoxic station-days, compare to the earlier DO-only logistic/historical-rate models on identical cases, and use event/time-block bootstrap for Brier skill/calibration/false alert tradeoffs. The earlier data supported only about 18 independent one-day severe onsets; a promising smooth from this phase alone is **not** sufficient for an operational alert threshold.

GNN imputation remains sensitivity-only. FCM remains a conceptual association/scenario comparator. No causal or operational warning claims should be inferred from these fits.
