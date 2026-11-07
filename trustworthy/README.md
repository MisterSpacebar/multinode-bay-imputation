# Trustworthy Biscayne Bay baseline — Phase 1

This is an observed-only, retrospective QA/target/backtest workflow. It neither
assumes that every data gap is sensor fouling nor claims to diagnose the cause
of a hypoxia episode. It does not depend on a deployments.csv.

## Quick start

Run from the repository root:

```bash
python -m unittest discover -s trustworthy/tests -v
python -m trustworthy.quality --input-dir imputed_output --out-dir trustworthy_outputs
python -m trustworthy.backtest --daily trustworthy_outputs/daily_targets.csv
```

The quality command accepts existing exports without running GNN training.
**For the new source-separated station identity, first retrain/re-export:**

```bash
python train.py --train
python train.py --impute --checkpoint checkpoints/best.pt
python -m trustworthy.quality
python -m trustworthy.backtest
```

These commands are separate, because the first requires model dependencies and
private local raw sensor files. The synthetic audit tests only require numpy
and pandas, and do not require those files.

## What changed

- No deployment metadata needed: backtests use the independent exported node.
  Supply optional `--deployments` and `--require-verified` only if known
  instrument epochs must be enforced. Lack of documentation does NOT
  disqualify a station from normal analysis.
- `preprocess.build_dataset()` defaults to `preserve_sources=True`. Separate
  2025 platform, 2026 platform, and buoy sources become distinct nodes, even
  if they share coordinates. Historical 3 km buoy matching and 100 m platform
  matching remain available only with
  `preprocess.build_dataset(preserve_sources=False)` for compatibility.
  Those legacy mergers are **not** valid station-level comparison data.
- `quality.py` excludes unobserved and out-of-physical-range DO from targets,
  but **flags rather than automatically removes** flatlines, isolated spikes,
  and zero readings. A 0 mg/L reading may be genuine severe anoxia.
- `events.py` logs a gap from 30 minutes, an outage from 2 hours, and an
  extended outage from 24 hours (all configurable in `EventConfig`).
  Gaps are detected between the first and last available export timestamps:
  start/end-of-deployment dates cannot be inferred from missing files.
- `targets.py` reports contiguous observed DO below 4 and 2 mg/L,
  including whether a missing boundary truncates the episode.
- All baseline forecast features derive from historical *observed-only*
  daily targets. No interpolated FCM weather or nutrient values are used.
  Training targets must end strictly before each forecast issue date.
- Daily labels require 95% valid DO coverage by default. Below that, labels
  become NaN, NOT zero. Days with 95–100% coverage report the counted observed
  low-oxygen hours: these are **lower bounds**, not estimated full-day hours.
- Sensor-quality event flags are separate from environmental low-DO events.
  High levels of suspect readings are a reason to investigate, not proof
  of instrument failure.

## Optional issue annotations (not a deployment file)

Sensor failures may result from debris, maintenance, fouling, power loss,
telemetry outages, or other causes. The model cannot identify the cause from
missing data. If verified operational notes exist, you may add a separate,
private review file (never necessary to run the analysis):

```csv
node,start_utc,end_utc,manual_label
2026_platform__station_A,2026-06-01T00:00:00Z,2026-06-02T00:00:00Z,maintenance_confirmed
```

These are **illustrative names and dates**, not factual station records.

```bash
python -m trustworthy.quality --manual-events water_data/sensor_issues.csv
```

All annotations remain user-declared, are exported as separate events, and
do not erase observations or alter the trained model.

Optional legacy `--deployments` can still record documented instrument
epoch transitions, but it is no longer required for quality checks or
forecast comparisons.

## Generated outputs (ignored by Git)

| File in trustworthy_outputs | Content |
|---|---|
| `station_audit.csv` | Per-station observation/invalid/suspect counts |
| `sensor_events.csv` | Outages, gaps, suspect flatlines/spikes, annotations |
| `daily_targets.csv` | Observed-only daily DO coverage, minima, hours below 4 and 2 |
| `hypoxia_events.csv` | Contiguous observed low-DO episodes, gap-censor flags |
| `backtest_predictions.csv` | Individual predictions with issue/target dates |
| `backtest_summary.csv` | MAE, RMSE, and persistence-relative skill |

## Important scientific qualifications

1. The raw historical merged outputs cannot be 'unmixed'. Retrain and
   regenerate the exported station series for source-separated evaluation.
   Source names and node count change; GNN checkpoints trained on the old
   graph cannot safely be reused with the new graph.
2. Within-source file groups can still include changes in calibration,
   deployment, instrument, or depth. The source separation does not prove
   physical station continuity; compare across documented transitions
   cautiously. A change of node is not proof of spatial independence.
3. This phase does not identify seasonal drivers, estimate nutrient loads,
   or predict future oxygen events probabilistically. Those are later stages.
4. The GNN is a **retrospective imputation model**. Its contemporary forcing
   and reconstruction windows are not a real-time future-forecast design.
5. Weather forcing in the existing root GNN may be filled using retrospective
   day-of-year climatology. Do not treat its imputation scores as operational
   forecast skill.
6. No real-world performance improvement is claimed until tests pass and
   outputs are checked against private local data.
