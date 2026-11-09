# Phase 2: observed-only state transitions and block uncertainty

This module extends Phase 1 episode extraction. It describes the **daily oxygen-state process**, not a new forecast model, and does not modify existing episode or QC definitions.

## Run from the repository root

```bash
python -m trustworthy.quality --input-dir imputed_output
python -m trustworthy.regimes
python -m trustworthy.episode_dynamics
python -m trustworthy.episode_transitions \
  --regimes trustworthy_outputs/onset/regime_days.csv \
  --episodes trustworthy_outputs/episode_dynamics/episode_table.csv \
  --out-dir trustworthy_outputs/episode_dynamics \
  --bootstrap 2000 --blocks 14day 28day month --seed 2026
```

The `--episodes` flag is optional. Without it, state and transition outputs are still written; onset-count uncertainty is omitted. All generated outputs live under `trustworthy_outputs/` and are not versioned measurement data.

## Scientific definition

- **Normal:** classifiable eligible day with no observed DO sample below 4 mg/L.
- **Moderate:** eligible day with an observed sample below 4 mg/L but none below 2 mg/L.
- **Severe:** eligible day with an observed sample below 2 mg/L.
- **Unknown:** either threshold lacks valid eligible DO coverage, or a date is absent. Unknown is not evidence of normal oxygen or recovery.
- A transition requires **two adjacent calendar dates at the same station**, both classifiable at 95% coverage (or the configured minimum).
- An exact concentration of 2.0 mg/L does not count as below 2; an exact 4.0 mg/L does not count as below 4. These strict inequalities originate upstream in `quality.py`.
- State classification is based on any observed five-minute threshold crossing that day, **not** a continuously low 24-hour profile. These are retrospective states.
- Repeated regional forcing, GNN reconstructions and other inferred values are never used to determine states.
- `qc_suspect_pair` flags transitions touching days with suspect QC observations; the underlying observed readings are retained according to the upstream policy.

The season of a transition is the **source-day** season. For example, August 31 to September 1 is recorded as JJA.

## Output files

| File | Description |
|---|---|
| `station_day_states.csv` | Every local station-day, including Unknown, with both threshold-hour fields and provenance |
| `observed_transitions.csv` | One row per genuine adjacent, eligible station-day pair |
| `transition_matrix_pooled.csv` | Pooled count and conditional probabilities, all nine state pairs |
| `transition_matrix_by_station.csv` | Per-station matrices, including zero-denominator rows |
| `transition_matrix_by_season.csv` | Season-of-source-date matrices (descriptive, not season-adjusted) |
| `transition_bootstrap.csv` | Pooled and station-specific approximate 95% block-bootstrap intervals, replicate denominators and counts |
| `onset_count_bootstrap.csv` | Optional bootstrap of identifiable onset **counts** from Phase 1, by station/threshold |
| `transition_metadata.json` | Input hashes, parameters, observed counts and assumptions |
| `figures/transition_heatmap.png` | Optional pooled-state heatmap (matplotlib required) |

A zero row denominator yields an undefined (`NaN`) probability rather than a zero probability. Station-specific uncertainty can be undefined for sparse states. The bootstrap marks such intervals as missing if fewer than 80% of resamples have a nonzero source-state denominator.

## Uncertainty design and limits

The bootstrap resamples the full set of **already-observed source-date transitions** within 14-day, 28-day or calendar-month station/time blocks, drawing each station's observed blocks with replacement. It does **not** splice sampled calendar blocks into new time series or create synthetic boundary transitions. Relative block schemes are sensitivity checks, not independent confirmatory experiments. These intervals remain approximate because serial dependence across blocks and synchronized events across stations are not fully reproduced by station-stratified sampling.

The onset-count bootstrap is conditional on the set of observed station-blocks. Its count interval is descriptive variability of repeated observation blocks, **not** a confidence interval on bay-wide incident event rates or a prospective probability.

Pooled transition estimates blend stations with different availability, observation seasons and dynamics. Always report source-state denominators and station matrices alongside pooled probabilities. Seasonal matrices are similarly unadjusted and cannot establish a climatological seasonal effect.

The Markov chain is **not** fitted in this module. Phase 3 will add right-censoring-aware duration analysis and a first-order Markov baseline, with recovery-hazard modeling as a separate subsequent step.

## Reproducibility / validation

```bash
python -m unittest discover -s trustworthy/tests -v
python -m py_compile trustworthy/*.py trustworthy/tests/*.py
```

The research report's observed-only daily input produces **1,178 known station-days** and **1,103 adjacent valid transitions**. Its reference matrix is:

| Origin | Normal | Moderate | Severe |
|---|---:|---:|---:|
| Normal | 265 | 68 | 4 |
| Moderate | 59 | 311 | 59 |
| Severe | 7 | 48 | 282 |

These are **input-file-specific regression targets**, not hardcoded constants or guaranteed across updates. When counts differ, inspect the station membership, dates, upstream QC/coverage, regime input hash and deployed source changes before concluding the method is incorrect.

Phase 2 does not alter Phase 1's observed-only episode table or derive imputed episodes. Its outputs are the baseline for Phase 3 survival and Markov uncertainty.
