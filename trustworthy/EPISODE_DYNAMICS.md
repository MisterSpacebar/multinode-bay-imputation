# Observed-only episode dynamics (Phase 1)

The module creates canonical descriptive daily episode records from the existing coverage-qualified observed-only regime outputs. It does not introduce a forecasting model.

## Run

    python -m trustworthy.quality --input-dir imputed_output
    python -m trustworthy.regimes
    python -m trustworthy.episode_dynamics --regimes trustworthy_outputs/onset/regime_days.csv --out-dir trustworthy_outputs/episode_dynamics

For precise, separately defined five-minute event detail, add --exports-dir imputed_output. Exports are read through trustworthy.quality.read_station, requiring observed flags, and source manifests when present. GNN reconstructions never replace missing observations.

## Definitions

- Daily hypoxic state: at least one valid measured DO sample strictly below 2 or 4 mg/L, and at least 95% coverage.
- Daily run: consecutive eligible local calendar days classified hypoxic. Unknown dates end a run and censor its boundary.
- Identifiable onset: immediately preceding local day is eligible and nonhypoxic. Starts at record boundaries or after unknown days are left-censored.
- Observed recovery: next local day is eligible and nonhypoxic. Ends at boundaries or before unknown days are right-censored, not recovered.
- Calendar duration: number of consecutive eligible low-oxygen days, NOT continuous hours of hypoxia.
- Cumulative low-oxygen exposure: observed hours below threshold, a lower bound on true exposure where coverage is less than 100%.
- Five-minute episode: contiguous valid observed low-DO samples (no gap bridging), detected with existing trustworthy.targets.detect_hypoxia_events. One daily run may include many five-minute episodes.
- Exactly 2.0 mg/L is not severe; exactly 4.0 mg/L is not below 4.
- Zero, flatline and spike-suspect samples remain measurements unless hard-invalidated by existing QC. Suspect flags are retained rather than silently filtered.

Start-to-start recurrence and fully observed nonhypoxic days between episodes are separate metrics; unavailable onset or recovery boundaries result in missing values.

## Outputs

- episode_table.csv: daily run-level details, censoring, exposure and recurrence
- station_episode_summary.csv: counts, observed recoveries and descriptive duration
- episode_summary.csv: threshold totals
- episode_season_summary.csv: exploratory event counts, NOT seasonal risk estimates
- analysis_metadata.json: input hash and method parameters
- figures: duration distribution and station counts, if matplotlib installed
- subdaily_episodes.csv (optional): exact measured-event bounds, DO minimum and mean, observed sample count, oxygen deficit and QC flags

The daily and five-minute tables use different definitions and MUST NOT be joined by episode ID.

Kaplan-Meier, recovery hazard, transition uncertainty and Markov modeling are intentionally deferred to later PRs. Deployment and sensor-identity provenance must be audited from upstream quality outputs before station-level scientific claims; station names alone cannot prove instrument continuity.

## Validate

    python -m unittest discover -s trustworthy/tests -v
    python -m py_compile trustworthy/*.py trustworthy/tests/*.py

The research report's 110 runs at <4 (72 identifiable onsets) and 81 runs at <2 (63 identifiable onsets) are validation targets for the SAME historical regime data version, never hardcoded constants. The prospective forecasting sample is smaller because it uses additional history/horizon and model-eligibility requirements. If outputs differ, check the input version, date range, station list, coverage, and source provenance before judging correctness.

Any later gap-bridging or imputed sensitivity analysis must produce a distinctly named output and may not overwrite the canonical table.
