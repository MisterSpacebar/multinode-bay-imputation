# Severe-hypoxia transition-warning prototype

This prototype **does use Markov chains and Monte Carlo**, alongside two regularized logistic prediction models. All three models are evaluated by chronological hindcasting. The target is the **first occurrence of severe hypoxia (at least one observed DO < 2 mg/L sample)** anywhere in the next 1, 3 or 7 local days, beginning from an eligible **Normal** or **Moderate** current day.

The prediction is made at **end of current day**; it is not a real-time alert. Already-Severe origins are deliberately excluded from new-severe-onset warnings; persistence/recovery for existing Severe episodes are covered by Phase 3.

## Inputs

Run prior validated modules to create:

- `trustworthy_outputs/daily_targets.csv` (Phase 1 QC, source-separated observations only)
- `trustworthy_outputs/episode_dynamics/station_day_states.csv` (Phase 2, daily N/M/S/Unknown)
- `trustworthy_outputs/episode_dynamics/observed_transitions.csv` (Phase 2, consecutive eligible station-day transitions)

If the historical Phase 3 run created an alternate output folder, pass these paths explicitly. No GNN reconstruction may be substituted for an observed measurement. Both threshold series must be eligible, at the required DO coverage (default 95%), to classify a state.

## Run from the repository root

```bash
python -m trustworthy.transition_warning \
  --daily trustworthy_outputs/daily_targets.csv \
  --states trustworthy_outputs/episode_dynamics/station_day_states.csv \
  --transitions trustworthy_outputs/episode_dynamics/observed_transitions.csv \
  --out-dir trustworthy_outputs/transition_warning \
  --min-coverage 0.95 \
  --min-train 60 --min-events 3 \
  --alpha 10 --station-prior 20 \
  --monte-carlo-paths 1000 --seed 2026
```

Disable optional simulation with `--monte-carlo-paths 0`.

## How the models work

1. **Station-shrunk Markov baseline.** The 3×3 Normal/Moderate/Severe transition matrix is estimated from historical **adjacent observed** transitions. Each station's transition counts are gently shrunk toward the past pooled matrix (`--station-prior 20`), with small pseudocounts to avoid zero probabilities. The probability of *at least one Severe day* during the next H days is computed **exactly** from the matrix: `1 - (Q^H 1)[origin]`, where Q is the Normal/Moderate submatrix. This is a stationary first-order **benchmark**, not a fully environmental state-space model.
2. **DO-history logistic model.** Standardized ridge logistic regression with station intercepts using *current-day observed* minimum/mean DO, observed hours below 2 and 4 mg/L, strict previous-calendar-day mean-DO change and current Moderate indicator.
3. **Environmental logistic model.** Same features plus observed same-day temperature, salinity and annual sine/cosine seasonal terms. It tests the **incremental value** of environmental measurements on the *same hindcast cases*.

The logistic models produce separate 1/3/7-day risk values. A deterministic within-issue cumulative maximum enforces the logical requirement that seven-day risk cannot be less than three-day risk, which cannot be less than one-day risk, when those predictions are present.

**Monte Carlo** samples 1,000 hypothetical seven-day Markov paths for eligible held-out station/month scenarios, and reports simulated chance of entering Severe within 1/3/7 days, severity-days in the seven-day path, and conditional first-severe timing. Results are compared with exact Markov probabilities for a sanity check. Simulations do **not** use GAM forecasts, do not add new observed data, and are not the basis of the logistic predictions.

## Hindcast/validation safeguards

- **Historical replay by calendar month:** for each test month, learned-model training uses only windows whose entire outcome period ended **before** that month. The Markov matrix also uses only transitions whose destination date is **before** the test month.
- **Horizon completeness:** all next 1/3/7 calendar days must have eligible observed states to label the window; a gap invalidates the whole evaluation window, even if Severe appeared before the gap.
- No future DO, temperature or salinity enters issue-day features. No missing-day forward-fill or GNN substitution.
- Markov, DO-only and environmental forecasts are issued on **identical complete-case** samples. Insufficient history/rare-event folds are skipped and recorded.
- 1/3/7-day labels are overlapping and correlated; pooled station-days are not independent observations. Overlapping labels count *forecast origins*, not distinct episodes.
- `25%` is a **demonstration warning threshold**, not a scientifically calibrated operational cutoff. Detection (recall), precision and the fraction of alerts that were false are descriptive diagnostics.
- If future observations are not yet available, the current code does not emit a deployed or live forecast; it implements a retrospective, auditable hindcast.

## Generated outputs

| File | Description |
|---|---|
| `warning_windows.csv` | Every non-Severe historical issue day × horizon, target label and exclusion reason |
| `hindcast_predictions.csv` | Out-of-sample probabilities and observed labels for three matched models |
| `hindcast_scores.csv` | Brier, Brier skill vs Markov, sample counts and illustrative 25% alert metrics |
| `fold_diagnostics.csv` | Monthly training/test counts and reasons for skipped folds |
| `monte_carlo_reference.csv` | Hypothetical sampled Markov trajectories and exact first-hit probabilities |
| `warning_metadata.json` | SHA-256 source hashes, settings, risk definition and limitations |
| `figures/hindcast_3day_example.png` | Historical risks and retrospectively observed severe events for one station |

## Synthetic validation

```bash
python -m unittest trustworthy.tests.test_transition_warning -v
python -m unittest discover -s trustworthy/tests -v
python -m py_compile trustworthy/*.py trustworthy/tests/*.py
```

The previously verified Phase 2 station-day and transition data contain 1,103 valid transitions; with those states alone, the candidate window builder produces **766 complete one-day, 689 complete three-day and 552 complete seven-day windows**. These are pre-model feature exclusions and therefore **not the final scored hindcast sample sizes**. There are 63 severe transitions among the 766 non-Severe one-day origins. Counts are historical-input-specific regression targets, not hardcoded outputs.

The next step is to run the command against the **real locally generated `daily_targets.csv`** and inspect scored cases, month-by-month variation, false alerts and improvement (or lack thereof) over Markov/DO-history. Do not interpret simulated risk estimates or in-sample fit as a validated operational warning system.
