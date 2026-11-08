# Time-valid environmental forcing and onset uncertainty (pilot)

Two new workflows complement the existing observed-only onset models without
changing their predictions or retrospectively importing future measurements.

## 1. Uncertainty on real onset-risk predictions

After `python -m trustworthy.onset`:

```bash
python -m trustworthy.onset_uncertainty
python -m trustworthy.onset_uncertainty --bootstrap 5000
```

The file `trustworthy_outputs/onset/onset_uncertainty.csv` compares, on
**identical station, issue, threshold and horizon windows**:

- DO logistic versus station historical-rate reference
- DO+temperature/salinity logistic versus historical rate
- Physical logistic versus DO-only logistic

For each pair, the report includes counts of positive *forecast windows*
and distinct observed onset dates, mean Brier and difference in Brier,
Brier skill, and exploratory 95% percentile intervals. All historical
outcome windows used to fit models end strictly before forecast issuance.

Resampling schemes are station-stratified month, 14-day and 28-day fixed
calendar blocks. They retain within-block serial dependence but do not
capture across-block and cross-station correlation; results with too few
blocks are withheld. Intervals are **approximate** and do not adjust for
post-hoc model selection, multiple comparisons or a small number of
independent hypoxia episodes. The longer-window resampling is a
sensitivity analysis, not a guarantee of valid prospective uncertainty.

## 2. Rain, wind and water level without looking into the future

`trustworthy/time_valid_forcing.py` takes a **local CSV of actual daily
observations with explicit provenance and release times**. We do not know
the original publication timestamp for historic MIA airport reports
from the ordinary date-only rainfall files. The script does not invent
those timestamps and does not download or modify sensor data.

**Each CSV row** is one station and one local calendar date; required
columns and units:

| Column | Required meaning |
|---|---|
| `node` | Exact station node, or `regional` for explicitly shared weather |
| `date` | Local calendar day, YYYY-MM-DD |
| `source_id` | Non-sensitive origin identifier (for traceability) |
| `rain_mm` | Daily total rainfall in millimeters |
| `wind_mps` | Daily mean wind speed in meters/second |
| `water_level_m` | Daily mean water level in meters, with consistent datum |
| `<field>_observed_end_utc` | UTC endpoint of the source measurement interval |
| `<field>_available_at_utc` | UTC time this value was available to the forecasting workflow |
| `<field>_coverage` | Fraction of nominal calendar-day observations (0–1) |

All three fields have their own `observed_end_utc`,
`available_at_utc` and `coverage`. Empty measurements must also have
empty corresponding provenance columns. Include every field's columns
even if wind or water level data are not yet available. Never set a
fictitious source publication timestamp and call it verified.

**At least 90% source coverage** is needed by default; there is no
forward/backward fill. If a day's value arrived after the forecast
issue cutoff it is withheld, even when its observation date is earlier.

### Lagged features

Run:

```bash
python -m trustworthy.time_valid_forcing \
    --forcing water_data/forcing_daily.csv \
    --lags 1,2 \
    --allow-regional-weather
```

The report `trustworthy_outputs/onset/time_valid_forcing.csv` includes
issue-date `rain_mm_lag1`, `rain_mm_lag2`, and corresponding lagged
wind/level fields where they actually exist and were released in time.
**Regional values are not broadcast by default**; even with
`--allow-regional-weather`, only rainfall and wind may be shared.
Water level always remains station-specific. Station-specific
measurements are never replaced with an easier regional value solely
because the local measurement is late or fails QC.

For MIA GHCN-Daily precipitation, use a **two-complete-day lag** as a
conservative provisional test pending verified reporting/release times.
A one-day lag is available for an explicit sensitivity scenario where
source timing supports it; it is not automatically declared operationally
available. If actual availability timestamps are unobtainable, do not
substitute arbitrary ones or claim operationally valid weather features.

### Optional onset model ablation

Only after a validated local forcing CSV exists:

```bash
python -m trustworthy.onset_forcing \
    --forcing water_data/forcing_daily.csv \
    --allow-regional-weather
```

This compares logistic models using `logit_physical` predictors plus:

- `logit_rain_lag2`: 2-day lag rainfall
- `logit_weather_lag2`: rainfall and wind at the same lag
- `logit_hydro_lag2`: rainfall, wind, and station-specific water level

All models are fitted on previously **completed** onset outcome windows;
predictions are only issued where input values actually exist and have
been published. There is no synthetic weather. Reports under
`trustworthy_outputs/onset/forcing/` include per-model exclusions,
probabilities, Brier/AP summaries and candidate-vs-existing-benchmark
Brier skill on **identical issue dates**. There is no promise that any
station has the necessary forcing coverage.

The forcing scripts use no externally hosted data connectors. Private
instrument metadata remains in `water_data/` and generated outputs
remain in `trustworthy_outputs/`, both gitignored.

## Limitations

Historical forecast issuance is defined as the end of the local day,
not an arbitrary real-time clock instant. Aggregate observations from
that day cannot enter the model unless the exact source publication time
precedes issuance. Separate preprocessing provenance and a further
prospective replay are needed before making operational alert claims.

The current 2 mg/L onset evaluation reuses a small number of distinct
episodes across overlapping windows. The new uncertainty intervals
measure uncertainty under chosen time blocks, not independent events.


## Final forcing builder QC / provenance update

The repository now provides `python -m trustworthy.build_forcing`, which
reads NDBC Virginia Key six-minute wind observations, follows them with
CO-OPS Virginia Key wind readings, downloads/caches CO-OPS station
8723214 water level (meters relative to **MSL**), and merges daily Miami
International Airport precipitation totals (inches converted to mm).
It writes `water_data/forcing_daily.csv` for stations in observed-only
daily targets. The forcing data remains **regional**, even though it is
repeated per station to match existing model input keys.

### Corrected wind source handoff

The builder now combines wind samples **by UTC timestamp before aggregating
by local Miami calendar date**. NDBC readings take precedence up to the last
valid NDBC sample; CO-OPS samples are used afterward. The CO-OPS request
starts on the UTC date containing the last NDBC sample, including any
remaining hours that belong to the preceding local date. Overlap timestamps
are deduplicated, and the affected local day's provenance records both
source names, e.g. `ndbc_vakf1+coops_8723214_wind`. This avoids dropping
a partial local day around the 2025/2026 New Year boundary.

### Explicit per-variable provenance columns

The builder retains the original field names expected by
`time_valid_forcing.read_forcing` and appends:

| Field suffix | Meaning |
|---|---|
| `_source` | Direct observation archive/source for that day's variable |
| `_spatial_scope` | `regional_proxy`, **not** a node-local sensor |
| `_coverage_basis` | `observed_six_minute_slots` for wind/level or `daily_record_present_not_subdaily_verified` for rain |
| `_availability_basis` | `assumed_fixed_lag_not_verified` |
| `_role` | `regional_meteorology` or `coastal_tidal_boundary` |

`rain_mm_coverage=1.0` is retained for compatibility with the current
forcing validator. It only means that a daily archival record exists;
**it does not assert that within-day rainfall sensor completeness was
independently measured or that rainfall was available in real time**.
Wind and water level completeness instead use the fraction of expected
six-minute slots, accounting for daylight-saving 23-/25-hour days.

The water level is a coastal tidal-boundary proxy, **not** the canal or
bay's site-specific water level. Its numeric datum is MSL; do not mix it
with NAVD88/NGVD29 station values without verified datum transformations.
Shared regional wind/level are not independent replications across 17
stations.

All `available_at_utc` values remain explicit fixed-lag assumptions:
3 hours after observation end for wind/water level and 24 hours after the
local daily rainfall period by default. No source-reported publication
timestamp is provided or implied. Do not upgrade the retrospective
forcing experiment to an operational forecast claim on these timestamps.

### Rebuild and verify

```bash
python -m unittest discover -s trustworthy/tests -v
python -m trustworthy.build_forcing
python -m trustworthy.time_valid_forcing --forcing water_data/forcing_daily.csv
python -m trustworthy.onset_forcing --forcing water_data/forcing_daily.csv
```

The source-file downloads and cache are local under
`water_data/coops_8723214/` and all generated forcing files remain
gitignored. Rebuild the daily forcing CSV after merging this fix to
apply the corrected local-day wind splice and provenance fields.
**Retraining the GNN and the FCM is not necessary.** The goal is
auditable environmental inputs, not new model fitting or hyperparameter
optimization. Preserve the current negative onset-forecast ablation
finding as part of the project conclusions.
