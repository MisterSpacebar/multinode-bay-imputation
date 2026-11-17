# Retrospective warning-policy and episode-level evaluation

This step **evaluates** the existing observed-only transition-warning hindcasts
(`trustworthy.transition_warning`). It does not retrain a warning model, add
predictors, or produce a forecast. Everything is **retrospective hindcasting**: no
operational calibration, no live validation and no causal claim.

It answers four questions on the *same matched hindcast cases*:

1. Do the models beat a **time-split** (held-out) climatology reference?
2. Which **cost-sensitive alert policy**, with thresholds chosen only from earlier
   out-of-sample predictions, is worth its false warnings?
3. Do alerts actually precede **distinct severe onsets**, and with what lead time?
4. How does onset detection trade off against **alert burden**?

## Inputs

Created by earlier phases (see `TRANSITION_WARNING.md`, `EPISODE_DYNAMICS.md`):

- `trustworthy_outputs/transition_warning/hindcast_predictions.csv`
- `trustworthy_outputs/transition_warning/warning_windows.csv`
- `trustworthy_outputs/episode_dynamics/episode_table.csv`
- `trustworthy_outputs/episode_dynamics/station_day_states.csv`
- `trustworthy_outputs/episode_dynamics/observed_transitions.csv`

## Run from the repository root

```bash
python -m trustworthy.warning_evaluation
python -m trustworthy.episode_warning_evaluation
python -m trustworthy.warning_burden
python -m trustworthy.warning_evaluation_figures
```

Each stage reads the previous stage's CSVs from `trustworthy_outputs/warning_evaluation/`
(override with `--root` / `--out-dir` / `--eval-dir`). Useful options:
`--bootstrap N` (default 2000), `--seed` (default 2026), `--skip-prior-sensitivity`,
`--min-prior-cases` (default 50) and `--min-prior-class` (default 5).

## Outputs (`trustworthy_outputs/warning_evaluation/`, ignored by Git)

| File | Contents |
|---|---|
| `heldout_climatology.csv` | Matched cases with all three model probabilities plus held-out `clim_state` / `clim_uncond` references and `n_train` |
| `brier_scores.csv`, `brier_comparisons.csv` | Held-out Brier and AUC per model/reference and paired Brier differences with station-month cluster-bootstrap intervals |
| `policy_decisions.csv` | Per case, model and cost ratio: probability, theoretical and chronological thresholds, whether a chronological threshold was available, binary alerts, outcome label, onset date and outcome-window end |
| `policy_scores.csv`, `policy_comparisons.csv` | Alerts per 100 issue-days, recall, precision, false-positive rate, false-alarm fraction and cost per 100 for each policy, plus paired cost differences |
| `prior_sensitivity.csv` | Climatology-prior and Markov-station-prior sensitivity (secondary; the primary analysis uses prior strength 20) |
| `onset_accounting.csv` | For every qualifying onset and horizon, whether it is evaluable and, if not, why |
| `episode_detection.csv` | Per onset/policy: detection (any alert and one-to-one credit), first alert, lead time, number of qualifying alert days |
| `warning_clusters.csv` | Consecutive-day alert sequences with duration, whether they precede an onset, and the onset they are credited to |
| `episode_policy_summary.csv`, `episode_policy_comparisons.csv` | Detection, alert burden, clusters per 100 days, false-cluster fraction, repeated-alert fraction and lead time, with intervals, and paired DO-history minus climatology differences |
| `detection_burden.csv` | Descriptive detection-versus-burden curves (see caveat below) |
| `figures/detection_vs_burden.png` | Main figure |
| `*_metadata.json` | Input and output SHA-256 hashes, parameters, counts, exclusions, limitations, code version (git commit and dirty flag) |

## Definitions

- **Severe onset**: `threshold_mgL == 2` (strictly below 2 mg/L), `identifiable_onset`
  and not `censored_left`; the onset date is `start_date_local`. An onset after an Unknown
  day is not identifiable. Station identities are never merged.
- **Forecast target and cases**: at least one Severe day within the next H eligible days,
  issued on a Normal/Moderate day (see `TRANSITION_WARNING.md`). Cases are the station-date-
  horizon rows scored by all three models. Overlapping windows are autocorrelated, so
  these are **not independent events**.
- **Held-out climatology**: pooled origin-state event rate shrunk toward the unconditional
  training rate with prior strength 20 (the pipeline's existing value, not tuned), fitted
  per test month and horizon on complete-case windows whose `window_end` is **before** the
  month starts. It is compared on identical cases and gets the same nested-window
  post-processing as the models.
- **Chronological threshold**: for each test month, choose the threshold minimising
  `cost_ratio * FN + FP` over *earlier* out-of-sample predictions (outcome windows ended
  before the month), then freeze it for the month. Ties go to the highest threshold. With
  too little history (fewer than 50 cases or 5 events/non-events) the threshold is
  unavailable and those cases are excluded from the policy comparison and reported.
  Cost ratios 2:1, 5:1 and 10:1 (missed event : false warning) are evaluated for 1- and
  3-day horizons.
- **Alert burden**: alert days per 100 **policy-evaluable issue-days** (cases with a
  chronological threshold), not per observed station-day.
- **Warning cluster**: a run of alerts on consecutive calendar days at one station. A missing
  or non-alert day ends it. Each cluster is credited only to its earliest qualifying onset
  (one-to-one); an onset reached by two clusters is credited once. "False cluster" means a
  cluster with no qualifying onset.
- **Evaluable onset**: has at least one scored hindcast opportunity with an available
  threshold. All qualifying onsets are accounted for, with reasons for exclusion.

## Safeguards

- Training for climatology and thresholds uses only outcomes completed before the test
  month (a window ending on the first day of the month is excluded).
- Alerts use frozen chronological decisions; they are never re-optimised once an onset occurs.
- Repeated alerts for one onset count once; unavailable early-fold cases are neither alerts
  nor opportunities.
- Uncertainty is a seeded **station-month** cluster bootstrap. For episode metrics it
  resamples units and recomputes every statistic while holding the chronological
  policy fixed, so it does not include threshold-selection variability.

## Interpretation limits

- Retrospective hindcast only; no operational calibration, no live validation, no causal claim.
- Few station-months and events (the local export has 63 qualifying onsets, of which a
  subset is evaluable): intervals are wide and station-level results are not rankable.
- `detection_burden.csv` ranks **evaluated** cases by predicted risk within the evaluation
  sample. It describes ranking quality and is **not** an implementable alert policy.
- Higher detection is not automatically a more useful policy; compare against "always
  alert" and against climatology at equal burden.

## Regression check against a local export

The generated artifacts are not tracked, so CI exercises these modules on synthetic data
that is run through the real `transition_warning` hindcast. To check a local export against
the targets recorded for the 2025-03 to 2026-08 data (1,402 matched cases; 63 qualifying
onsets, 48 evaluable; 469 and 429 policy cases):

```powershell
$env:TRUSTWORTHY_REAL_OUTPUTS = "C:\path\to\trustworthy_outputs"
python -m unittest discover -s trustworthy/tests -p "test_warning_evaluation_pipeline.py" -v
```

These are regression targets for that dataset, not constants of the method.
