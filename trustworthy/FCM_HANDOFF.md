# FCM handoff: source-separated, observation-aware concept inputs

The existing FCM is an **exploratory descriptive model**, not yet a validated
forecast or causal model. The trustworthy daily QC and rolling-origin baseline
are the reference evaluation system. Do not report FCM skill based on its
in-sample fit or on normalized reconstruction loss.

## Confirmed integration hazards in the current root fcm.py

1. `IMPUTED_FILES` lists seven **old merged export names**. New GNN exports
   have 17 source-specific names; the existing loader cannot automatically
   find them. Do not recreate old merged names to make the code run.
2. `load_physical_concept_timeseries()` averages station values across sites
   and weights model-imputed readings at 0.5 even where no true observations
   exist. Values from distinct physical systems must not become an
   unqualified "bay-wide" observed daily series.
3. `.interpolate(method="time", limit=14).ffill().bfill()` brings later
   values into earlier dates. This is acceptable *only* for clearly labeled
   retrospective visualization, never for a forecasting feature matrix.
4. The nutrient FCM combines sparse 2021–2024 discrete surveys with
   high-frequency 2025–2026 sensor histories; nutrient concepts are missing
   in the continuous era. Monthly interpolation and cross-source aggregation
   must not be presented as observed nutrient trends.
5. Ridge coefficients normalized to [-1,1] and applied under a sigmoid FCM
   simulation do not have the same objective as the regression that learned
   them. Forecast and scenario/interpretation components must be separate.

## Minimal Phase 2 architecture

- **Observed concepts**: station-day `odo_mean`, `odo_min`,
  `hours_lt4`, `temp_mean`, and `sal_mean` from
  `trustworthy_outputs/daily_targets.csv`. Preserve `node`,
  `date`, `coverage`, `eligible`, `identity_caution`.
  Do not interpolate daily targets or use GNN imputed values as ground truth.
- **Exogenous drivers**: rainfall, air temperature, wind, tide and/or
  discharge with explicit UTC observation timestamps and an "available by"
  date. Respect release timing; full-day weather is only usable in a forecast
  issued *after* that day.
- **Physical FCM v2**: start with 3 measured concepts (DO, water temperature,
  salinity), then expand only when there is observed coverage. Use
  station-specific or explicitly labeled regional summaries, never average
  distant bay/canal instruments by default.
- **Descriptive map**: allow expert-specified directed signs and sparse
  estimated weights, and present links as hypotheses/associations.
- **Predictive map**: calibrate on strictly past training dates, forecast
  held-out target dates and compare to paired persistence and Ridge.
  No retrospective interpolation into the evaluation period.
- **Nutrient FCM**: a separate sensitivity/interpretation experiment. Keep
  sparse grab-sample survey locations and sample depth, and label concepts
  as missing outside their observed periods.

## Promotion criteria

Proceed with GNN-FCM fusion only after (1) per-node coverage diagnostics
identify the available sample and low-oxygen event counts; (2) at least one
time-valid forcing covariate can be assembled at forecast issue dates; (3)
FCM-only performance is reported on the same issue/target dates as the
trustworthy baseline, including event-conditioned errors; and (4) the link
weights remain interpretable without inferring causation from associations.

A separate FCM-focused PR should replace the seven hardcoded inputs and
leaky interpolation before producing any new scientific FCM forecast claims.
