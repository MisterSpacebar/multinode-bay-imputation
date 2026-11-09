# Phase 3 — Episode persistence, recovery and Markov baseline

This stage extends the observed-only Phase 1 episodes and Phase 2 station-day states/transitions. It does **not** create imputed episodes, causal effects or operational forecasts.

## Reproduce

Run the previous pipeline first:

    python -m trustworthy.quality --input-dir imputed_output
    python -m trustworthy.regimes
    python -m trustworthy.episode_dynamics
    python -m trustworthy.episode_transitions --episodes trustworthy_outputs/episode_dynamics/episode_table.csv

The recovery GLM additionally requires optional dependencies:

    python -m pip install "statsmodels>=0.14,<0.16" "scipy>=1.11,<2"

Run:

    python -m trustworthy.episode_survival --episodes trustworthy_outputs/episode_dynamics/episode_table.csv --states trustworthy_outputs/episode_dynamics/station_day_states.csv --transitions trustworthy_outputs/episode_dynamics/observed_transitions.csv --out-dir trustworthy_outputs/episode_dynamics --bootstrap 2000 --blocks 14day 28day month --seed 2026

With only existing baseline numpy and pandas dependencies, add `--skip-hazard`. Plotting is optional (matplotlib). `--station-effects` fits a separate, unstable station-fixed-effects sensitivity model when estimable; no model is reported if a station has complete outcome separation. Use `--path-simulations 0` to omit stochastic paths, or `--save-markov-draws` to export per-replicate analytical Markov metrics.

## Scientific definitions

**Kaplan–Meier duration:** only identifiable observed onsets (previous day known, nonhypoxic) enter this population. Observed duration is **consecutive eligible hypoxic local calendar days**, not a continuous hour-by-hour excursion. Observed recovery requires an adjacent eligible nonhypoxic calendar day. Unknown next days, missing dates, and record boundaries are **right-censored**. Left-censored starts are excluded, and the 2/4 mg/L threshold samples are nested, not independent. Greenwood variance is a basic diagnostic and does not fix dependent station/episode observations or informative censoring. If survival does not fall to 0.5, the KM median is explicitly undefined.

**Next-day recovery hazard:** each row represents an observed onset-defined episode-day with *known next-day state*. Terminal days whose next-day state is unknown are excluded from the likelihood, never labeled negative recovery. The exploratory complementary-log-log binomial model uses current-day DO exposure hours per **six-hour increment** and log elapsed episode days. The next-day recovery outcome is distinct from current-day predictors. Standard errors are sandwich-clustered by episode, not by station/weather date; serial/station dependence may remain. Pooled regression may confound station. Station-fixed-effects is optional and can be nonidentifiable for sparse stations. No GNN values, rainfall, wind, salinity, temperature, or assumed source publication times enter this baseline. Model coefficients need not match earlier *weather-adjusted* exploratory research.

**Markov baseline:** resample complete 3×3 transition-count matrices by resampling station-specific source-date blocks (14-day, 28-day, month). Bootstrap blocks are observed transitions only; no artificial joins across unknown days and no independently sampled marginal probabilities. Each bootstrap matrix yields *analytical* expected consecutive Moderate/Severe runs, Severe→Normal first-passage expectation, and probability of reaching Severe in 7 transitions from Normal/Moderate. Monte Carlo trajectories optionally add path randomness and report recoveries/censoring at a fixed 120-day simulation horizon; they are model simulations, not new observations. Undefined/sparse bootstrap statistics are missing rather than invented. Pooling assumes stationary first-order transitions across stations, despite known heterogeneity and duration dependence.

## Outputs

- `survival_curves.csv`: pooled/station KM steps, recovery and censoring counts, Greenwood standard errors
- `survival_summary.csv`: onsets, recoveries, censor counts, observed median and KM median
- `recovery_risk_days.csv`: auditable next-day outcome and current-day exposure by episode
- `recovery_hazard.csv`: pooled episode-clustered cloglog estimates, if statsmodels installed
- `recovery_hazard_station_FE.csv`: optional station-effects exploratory sensitivity
- `markov_monte_carlo_summary.csv`: pooled Markov metrics with 14/28-day/month matrix-bootstrap uncertainty
- `markov_matrix_bootstrap_draws.csv`: optional analytical per-replicate metrics
- `markov_path_simulation_summary.csv`: optional simulated model recovery and horizon-censoring
- `survival_metadata.json`: SHA-256 input hashes, settings and scientific safeguards
- `figures/episode_km.png`: optional KM plot

## Research comparison targets

For the archived data version: 63 identifiable <2 onsets (46 observed recoveries, 17 censored), 72 identifiable <4 onsets (52 observed recoveries, 20 censored); KM median about **3 days** for both. The Phase 2 transition count matrix is `[[265,68,4],[59,311,59],[7,48,282]]` over 1,103 paired station-days. From the fixed matrix, expected consecutive severe days = approximately 6.13; expected Severe→Normal first passage = approximately 16.50 days. Resampling the matrix changes these statistics; the mean Markov run length cannot be equated to the observed median duration. These values are **version-specific validation targets**, not hardcoded constants or validated external forecasts.

Validation:

    python -m unittest discover -s trustworthy/tests -v
    python -m py_compile trustworthy/*.py trustworthy/tests/*.py

The next scientific step after reviewing this PR is an observed-only station-aware GAM/GAMM of DO severity and recovery, using verified local temperature/salinity and *separately* introducing regional forcing in ablations.