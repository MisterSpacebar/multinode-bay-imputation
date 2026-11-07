# Trustworthy water-quality baseline (v1)

This folder is deliberately separate from the legacy GNN and FCM analysis.
It does **not** change the original FCM results, make causal claims, or promise
operational hypoxia alerts.

## Why this exists

- The old GNN normalizer saw validation dates and its validation input was
  unmasked. The corresponding root-level \`train.py\` fixes fit the normalizer
  on training dates and score deterministic, deliberately hidden observations.
- The old \`preprocess.py\` forward-filled up to three five-minute samples,
  silently converting copied values into "observations". That fill is removed.
- Original station exports may combine distinct instruments. The optional
  deployment map prevents the trusted backtest from crossing instrument epochs.
- The old \`north_bay_holdout.py\` uses forcing derived from FCM interpolation.
  This new baseline **does not use FCM-derived forcing**.

## Commands (from repository root)

\`\`\`bash
python -m unittest discover -s trustworthy/tests -v
python -m trustworthy.quality --input-dir imputed_output --out-dir trustworthy_outputs
python -m trustworthy.backtest --daily trustworthy_outputs/daily_targets.csv
\`\`\`

The first command runs synthetic tests without private station data. The
second needs existing \`imputed_output/*_imputed.csv\` exports. The third needs
the daily target CSV produced by the second command.

To enable *verified* cross-time forecasting, provide a private CSV such as
\`water_data/deployments.csv\` with columns:

\`\`\`csv
node,start_utc,end_utc,deployment_id
station_A,2025-01-01T00:00:00Z,2025-12-31T00:00:00Z,epoch_1
station_A,2025-12-31T00:00:00Z,2027-01-01T00:00:00Z,epoch_2
\`\`\`

Use exact station filenames without \`_imputed.csv\` as node names.
The example is illustrative; **do not** assign dates or identities without
documentation. No real source names or locations need to be published.

\`\`\`bash
python -m trustworthy.quality --deployments water_data/deployments.csv
python -m trustworthy.backtest
\`\`\`

Without verified epochs, the trusted backtest intentionally produces no
comparisons. For *exploratory* results only:

\`\`\`bash
python -m trustworthy.backtest --allow-unverified
\`\`\`

## Semantics and limitations

- Timestamps are interpreted as UTC if the export has no timezone. Daily
  targets use America/New_York local dates and handle DST day length.
  Verify the input timezone before trusting results.
- Missing, out-of-range, and long exactly constant DO runs are not accepted
  as validated DO. Exact zero is flagged for review but not automatically
  deleted; genuine anoxia is possible.
- At least 95% valid 5-minute DO coverage is required by default. Missing
  periods are **not** assumed to have normal oxygen. \`hours_lt4\` and
  \`hours_lt2\` count *measured* hours and are lower bounds when gaps remain.
- \`predawn_min\` is the local 00:00–06:00 minimum with >=90% coverage.
  It is not necessarily the whole-night minimum.
- Missing temperature/salinity flags lead to NaN predictors rather than
  treating imputed covariates as observed.
- Forecasts are issued after the full issue day has completed. Targets are
  single future calendar days, not rolling multi-day aggregates.
- Model training for each forecast uses only targets that finished strictly
  before its issue date. Test targets are not used for fitting. All baselines
  use the same eligible samples; no extrapolated target values.
- The comparison uses 2025-09 and 2026-09 target months as predefined
  holdouts, with horizons 1, 3, and 7 days. These are **not** independent
  event-level validation, and the same deployment is required at issue/target.
- No hydrologic forcing is included yet: this is a trustworthy baseline,
  not the final scientific forecast. The Ridge model is a simple reference.
- The root GNN is an **imputation** model, not a forward forecaster.
  Its fixed validation mask tests point reconstruction, not block gaps,
  station transfer, or prospective predictions.
- Root \`train.py\` still uses contemporaneous daily weather and a graph
  derived from all stations. These may be appropriate for retrospective
  reconstruction but **not** an operational forecast at issue time.
- The root preprocessing currently merges some geographically nearby
  instruments. This new package records epoch uncertainty but cannot
  unmix already averaged readings; fixing upstream source-level identity
  remains a separate task.
- After changing root preprocessing, regenerate exports and retrain the GNN.
  Existing checkpoints and CSV exports are **not** retrospectively repaired.
- \`trustworthy_outputs/\` is ignored by Git. Keep private metadata in
  \`water_data/\`, also ignored by Git.

## Outputs

\`trustworthy_outputs/station_audit.csv\`: observation, invalid, flatline,
zero-review, and valid counts by exported node.

\`trustworthy_outputs/daily_targets.csv\`: observed-only daily labels,
coverage, eligibility, and deployment epoch.

\`trustworthy_outputs/backtest_predictions.csv\`: each issued prediction
with target date, horizon, actual, prediction, and errors.

\`trustworthy_outputs/backtest_summary.csv\`: station/month/horizon MAE,
RMSE, event-day counts and skill against persistence.

**Interpretation:** empty backtests are a legitimate warning about missing
provenance or insufficient training history, not evidence of zero risk.
