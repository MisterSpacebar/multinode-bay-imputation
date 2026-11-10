# Warning dashboard, station snapshots and console hindcast queries

This is the **presentation and usability layer** for the validated Phase 5
1/3/7-day Severe-hypoxia prototype. It does not replace that model or modify
the observed-only sensor QC, threshold states, or episode definitions.

All outputs are clearly labeled **retrospective research estimates**, not an
operational live warning service.

## Prerequisites

Generate these outputs using the existing repository pipeline:

- \`trustworthy_outputs/daily_targets.csv\` (observed-only DO, temp, salinity)
- \`trustworthy_outputs/episode_dynamics/station_day_states.csv\`
- \`trustworthy_outputs/episode_dynamics/observed_transitions.csv\`
- \`trustworthy_outputs/transition_warning/warning_windows.csv\`
- \`trustworthy_outputs/transition_warning/hindcast_predictions.csv\`

To regenerate the last two, run:

\`\`\`bash
python -m trustworthy.transition_warning \
  --daily trustworthy_outputs/daily_targets.csv \
  --states trustworthy_outputs/episode_dynamics/station_day_states.csv \
  --transitions trustworthy_outputs/episode_dynamics/observed_transitions.csv \
  --out-dir trustworthy_outputs/transition_warning
\`\`\`

If station files are in nonstandard locations, provide the corresponding
\`--daily\`, \`--states\`, \`--transitions\`, \`--windows\`, and \`--predictions\`
paths to the dashboard. On Windows PowerShell, run the command on one line
or use backtick (\` \`) continuations instead of Bash backslashes.

## Generate visualizations and tables

\`\`\`bash
python -m trustworthy.warning_dashboard \
  --out-dir trustworthy_outputs/transition_warning/dashboard \
  --threshold-grid 0.10 0.15 0.20 0.25 0.30 0.40 0.50
\`\`\`

Use \`--timeline-stations\` followed by **exact node identifiers** from
\`station_day_states.csv\` to select specific timeline panels, or accept the
three default stations with the most comparable 3-day hindcasts. Set
\`--max-timeline-stations 0\` to omit automatic timelines.

**Output CSVs:**

- \`warning_threshold_summary.csv\`: 1/3/7-day operating curves for **all
  three models** and each configured cutoff (recall, precision, false-positive
  rate, false-alarm fraction, F1, balanced accuracy, and alert volume)
- \`warning_threshold_selected_models.csv\`: highlighted **research-prototype**
  model choice: DO history at 1 and 3 days, Markov at 7 days
- \`calibration_summary.csv\`: predicted vs observed event rates in probability
  bins; bins with fewer than 10 cases are explicitly flagged
- \`latest_warning_snapshot.csv\`: the **latest eligible day *in each station's
  available data file***, its absolute date, stale-days relative to the newest
  station's data, current state, and 1/3/7-day model risk estimates
- \`dashboard_metadata.json\`: input content hashes and interpretation caveats

**Output PNGs (matplotlib required):**

- \`figures/threshold_tradeoff_1d.png\`,
  \`threshold_tradeoff_3d.png\`, \`threshold_tradeoff_7d.png\`
- \`figures/reliability_1d.png\`, \`reliability_3d.png\`,
  \`reliability_7d.png\`
- \`figures/hindcast_model_comparison.png\`
- \`figures/station_risk_heatmap_3d.png\`, \`station_risk_heatmap_7d.png\`
- \`figures/timeline_<station>.png\` — same-station observed daily minimum DO
  with severe-day markers plus 1/3/7-day selected-model warning probabilities

Station heatmaps contain **only dates with matched out-of-sample hindcasts**;
black crosses indicate days for which Severe hypoxia was subsequently observed
within the plotted forecast horizon. Empty areas are not “zero risk.”
Timelines are retrospective; no background imputation or invented oxygen
readings are used.

## Console query: historical date and recorded outcome

Copy the exact **node** from \`station_day_states.csv\` and choose a historical
end-of-day issue date (YYYY-MM-DD):

\`\`\`bash
python -m trustworthy.warning_dashboard \
  --query-node "2025_platform__raw-data-platformL1_parameters" \
  --query-date 2025-07-15
\`\`\`

The console displays one row per horizon and prediction model:

- Probability of **at least one Severe day** in the following 1, 3, or 7 days
- Suggested qualitative alert band
- Whether Severe **actually occurred** within the complete horizon, when
  all future observed days meet QC/coverage requirements
- Whether the prediction is a cached **monthly out-of-sample hindcast**, or a
  fresh strictly **as-of end-of-day refit** (using only already completed
  historical labels; never the queried day's future outcomes)

It then prints the recorded states and observed minimum DO for the next seven
calendar dates, marking unavailable days **Unknown**. It also saves
\`hindcast_date_query.csv\` and \`hindcast_actual_future.csv\`.

For a prompt-driven interactive console, run:

\`\`\`bash
python -m trustworthy.warning_dashboard --interactive
\`\`\`

The prompt lists valid station IDs. Type a station ID and YYYY-MM-DD, then
repeat, or press Enter at the station prompt to exit. To avoid regenerating
figures in interactive use, add \`--no-figures --no-snapshot\`.

**Important:** If the chosen date is not present in the matched hindcast
predictions, the module performs a new *as-of* fit for that historical date.
This remains leakage-safe, but it is not identical to an expanding
**month-boundary** hindcast: it may use more already completed information
than the monthly held-out evaluation did. The output explicitly indicates
which source applies.

If there is insufficient earlier history for a logistic fit, a historical
Markov baseline may still be available. If the station was already Severe on
the queried day, the new-onset risk is **not applicable**; consult the Phase 3
recovery/persistence outputs. Missing future days are not counted as
negative events.


### Compare all stations on one historical date

You can now submit a date **without selecting an individual station**:

\`\`\`bash
python -m trustworthy.warning_dashboard --query-all --query-date 2025-07-15 --no-figures --no-snapshot
\`\`\`

The equivalent option \`--all-stations\` also works. In the interactive console
(\`--interactive\`), enter \`ALL\` as the station name and then type the date.

The console prints **one row per station**, with each station's recorded
issue-day state, 1/3/7-day risk, and what was actually observed within each
complete future horizon. Source files include:

- \`hindcast_all_stations_summary.csv\`: per-station 1/3/7-day selected-model
  risks and observed Severe outcomes when known; the model choice and
  prediction source are explicit for each horizon
- \`hindcast_all_stations_models.csv\`: available Markov, DO-history and
  environmental risk estimates, with alert levels and outcome availability
- \`hindcast_all_stations_actual_future.csv\`: observed oxygen state and daily
  minimum DO over the next seven calendar days for every station

The station-wide query **does not average probabilities across the bay**.
It performs separate station-specific predictions based on the available
shared training data, and accounts for station-specific Markov transitions.
The output explicitly distinguishes stations already Severe, Unknown,
missing on the selected date, and ineligible or unsupported forecasts.
Where there is insufficient history for the selected 1/3-day DO-history model,
a Markov fallback may be shown and labeled as such.

A historic date can be eligible for some stations but not others; those
without adequate observations are included with missing probabilities, not
silently dropped. Future observations remain a separate post-forecast
comparison, never model training inputs.

## Latest-observation snapshot (not a live forecast)

The latest snapshot uses the **latest eligible recorded day per station**.
A station's last available observation can be months or years old. The snapshot
shows its date and \`stale_days\` relative to the newest data in the collection,
**not** relative to a real-time clock. \`already_severe\`, \`no_eligible_observed_day\`,
and \`insufficient_history_or_features\` receive distinct statuses.

For example, the spreadsheet-like output contains:

\`\`\`text
node,issue_date,current_state,stale_days,risk_1d,risk_3d,risk_7d,alert_level_3d,status
\`\`\`

**Not a live monitoring or alerting system.** No automated future data
ingestion, sensor sync, alert delivery, or operational calibration is performed.

## Alert thresholds and evaluation

Qualitative display bands are **illustrative**, not calibrated risk categories:

- Low: below 10%
- Watch: 10% to below 25%
- Elevated: 25% to below 50%
- High: at least 50%

For each candidate cutoff, the threshold sweep computes detection recall,
precision, false alarms, false-positive rate and number of issued alerts.
Changing the threshold does **not** refit the underlying statistical model.
Using the same hindcast period to select a favorite cutoff and to report its
performance is exploratory selection, not independent validation.

Reliability plots compare average predicted probabilities with observed rates;
small bins are unstable. The comparison across 1/3/7-day models involves
different complete-case populations, and overlapping windows are correlated.

## Validation

\`\`\`bash
python -m unittest trustworthy.tests.test_warning_dashboard -v
python -m unittest discover -s trustworthy/tests -v
python -m py_compile trustworthy/*.py trustworthy/tests/*.py
\`\`\`

This completes the intended **research visualization and date-query UX** for
the existing warning model. It does not claim weather-causal warning signs,
guaranteed event detection, or operational accuracy.
