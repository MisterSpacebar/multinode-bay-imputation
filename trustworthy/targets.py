"""Observed-only low oxygen episode detection.

Events are descriptive intervals with at least one measured DO value.
Gaps break episodes and adjacent missing readings censor their boundary.
Zero-oxygen measurements are retained. No imputed values are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_hypoxia_events(qc: pd.DataFrame, node: str,
                          interval_minutes: int = 5,
                          thresholds=(4.0, 2.0)) -> list[dict]:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    result = []
    if qc.empty:
        return result
    step = pd.Timedelta(minutes=interval_minutes)
    if qc.index.has_duplicates:
        raise ValueError("Duplicate QC timestamps")
    x = qc.sort_index()
    values = x["do_valid"]
    # Use observed-only QC DO. Missing/invalid values break the episode.
    for threshold in thresholds:
        active = False
        start = last = None
        minimum = np.inf
        left_censored = False
        previous_time = previous_valid = None

        def close(right_censored: bool):
            if not active:
                return
            result.append({
                "node": node,
                "threshold_mgL": float(threshold),
                "start_utc": start.isoformat(),
                "end_utc": (last + step).isoformat(),
                "observed_duration_minutes": int(
                    ((last - start) / step + 1) * interval_minutes),
                "min_odo_mgL": float(minimum),
                "censored_left": bool(left_censored),
                "censored_right": bool(right_censored),
            })

        for timestamp, val in values.items():
            present = bool(np.isfinite(val))
            continuous = (previous_time is not None
                          and timestamp - previous_time == step)
            is_low = present and float(val) < threshold
            if active and (not continuous or not is_low):
                close(not continuous or not present)
                active = False
            if is_low:
                if not active:
                    start = timestamp
                    minimum = float(val)
                    # A preceding missing or discontinuous sample limits
                    # certainty about the episode's true onset.
                    left_censored = (previous_time is None
                                     or not continuous or not previous_valid)
                    active = True
                minimum = min(minimum, float(val))
                last = timestamp
            previous_time = timestamp
            previous_valid = present
        if active:
            close(True)
    return sorted(result, key=lambda v: (v["start_utc"], v["threshold_mgL"]))
