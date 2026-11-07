"""Observation-derived sensor issue timeline. No deployments.csv required.

An automatically detected gap is not evidence of debris or maintenance.
Any manual reason is a separate, explicitly attributed annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventConfig:
    interval_minutes: int = 5
    gap_warn_minutes: int = 30
    outage_minutes: int = 120
    extended_outage_minutes: int = 1440

    def __post_init__(self):
        if self.interval_minutes <= 0 or self.gap_warn_minutes < self.interval_minutes:
            raise ValueError("Invalid sensor interval or gap threshold")
        if not (self.gap_warn_minutes <= self.outage_minutes
                <= self.extended_outage_minutes):
            raise ValueError("Gap thresholds must be in ascending order")


def load_manual_events(path: Path | None) -> pd.DataFrame | None:
    """Optional review log: node,start_utc,end_utc,manual_label.

    These events annotate sensor issues; they never silently erase values.
    """
    if path is None:
        return None
    df = pd.read_csv(path, dtype={"node": str, "manual_label": str})
    required = {"node", "start_utc", "end_utc", "manual_label"}
    if not required.issubset(df):
        raise ValueError(f"Manual event CSV needs {sorted(required)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Manual events have missing fields")
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="raise")
    df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True, errors="raise")
    if (df["end_utc"] <= df["start_utc"]).any():
        raise ValueError("Invalid manual event interval")
    return df


def _runs(mask: np.ndarray):
    """Return half-open intervals [start, end) of consecutive True values."""
    idx = np.flatnonzero(mask.astype(bool))
    if len(idx) == 0:
        return
    boundaries = np.flatnonzero(np.diff(idx) > 1) + 1
    for grp in np.split(idx, boundaries):
        yield int(grp[0]), int(grp[-1]) + 1


def _entry(node, event_type, start, end, severity, evidence,
           auto_detected=True, manual_label=""):
    return {
        "node": node,
        "event_type": event_type,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "duration_minutes": (end - start).total_seconds() / 60,
        "severity": severity,
        "evidence": evidence,
        "auto_detected": auto_detected,
        "manual_label": manual_label,
    }


def detect_quality_events(qc: pd.DataFrame, node: str,
                          config: EventConfig = EventConfig(),
                          manual: pd.DataFrame | None = None) -> list[dict]:
    """Detect missing-observation intervals, suspect flatlines and spikes.

    Only the interval from the first through last indexed data point is
    evaluated. No claim is made about pre-installation/post-last-record time.
    """
    events = []
    if not qc.empty:
        step = pd.Timedelta(minutes=config.interval_minutes)
        if qc.index.tz is None:
            raise ValueError("Sensor QC index must have timezone")
        if qc.index.has_duplicates:
            raise ValueError("Duplicate timestamps")
        grid = pd.date_range(qc.index.min(), qc.index.max(),
                             freq=step, tz="UTC")
        observed = qc["do_flag_observed"].reindex(grid).fillna(False).astype(bool)
        missing = (~observed).to_numpy(dtype=bool)
        for a, b in _runs(missing):
            length = (b - a) * config.interval_minutes
            if length < config.gap_warn_minutes:
                continue
            if length >= config.extended_outage_minutes:
                kind, severity = "extended_outage", "high"
            elif length >= config.outage_minutes:
                kind, severity = "outage", "warning"
            else:
                kind, severity = "short_gap", "review"
            events.append(_entry(node, kind, grid[a], grid[b - 1] + step,
                                 severity, f"{b-a} scheduled samples not observed"))

        for column, kind, severity in [
            ("do_flatline", "flatline_suspect", "review"),
            ("do_spike_suspect", "spike_suspect", "review"),
            ("do_invalid", "range_failure", "high"),
            ("do_zero_review", "zero_oxygen_review", "review"),
        ]:
            if column not in qc:
                continue
            flags = qc[column].reindex(grid).fillna(False).to_numpy(dtype=bool)
            for a, b in _runs(flags):
                events.append(_entry(
                    node, kind, grid[a], grid[b - 1] + step,
                    severity, f"{b-a} flagged 5-minute samples"))

    if manual is not None:
        for _, row in manual.loc[manual["node"] == node].iterrows():
            events.append(_entry(
                node, "manual_annotation", row["start_utc"], row["end_utc"],
                "reviewed", "User-supplied annotation",
                auto_detected=False, manual_label=str(row["manual_label"])))
    return sorted(events, key=lambda e: (e["start_utc"], e["event_type"]))
