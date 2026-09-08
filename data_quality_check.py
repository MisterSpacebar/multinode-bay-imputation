"""Audit imputed CSV outputs for physical ranges and observation coverage.

Run:
    python data_quality_check.py

The audit is intentionally read-only. It reports violations in the files as
currently written and writes analysis/data_quality_audit.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import PHYSICAL_BOUNDS


IMPUTED_DIR = Path("imputed_output")
OUTPUT_PATH = Path("analysis/data_quality_audit.csv")


def audit_file(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    rows = []
    for feature, (lower, upper) in PHYSICAL_BOUNDS.items():
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values.dropna()
        observed_col = f"{feature}_observed"
        observed = frame[observed_col].astype(bool) if observed_col in frame else pd.Series(
            True, index=frame.index
        )
        invalid = values.notna() & ((values < lower) | (values > upper))
        imputed_invalid = invalid & ~observed
        rows.append(
            {
                "file": path.name,
                "feature": feature,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_values": int(finite.size),
                "n_observed": int(observed[values.notna()].sum()),
                "n_imputed": int((~observed & values.notna()).sum()),
                "n_invalid": int(invalid.sum()),
                "n_invalid_imputed": int(imputed_invalid.sum()),
                "min": float(finite.min()) if not finite.empty else np.nan,
                "max": float(finite.max()) if not finite.empty else np.nan,
            }
        )
    return rows


def main() -> int:
    paths = sorted(IMPUTED_DIR.glob("*_imputed.csv"))
    if not paths:
        raise FileNotFoundError(f"No imputed CSV files found in {IMPUTED_DIR}")

    rows = [row for path in paths for row in audit_file(path)]
    report = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False, float_format="%.6g")

    invalid_total = int(report["n_invalid"].sum())
    invalid_imputed_total = int(report["n_invalid_imputed"].sum())
    print(f"Audited {len(paths)} files and {len(report)} feature/file pairs")
    print(f"Invalid values: {invalid_total} total, {invalid_imputed_total} imputed")
    print(f"Saved: {OUTPUT_PATH}")
    if invalid_total:
        print(report.loc[report["n_invalid"] > 0, [
            "file", "feature", "n_invalid", "n_invalid_imputed", "min", "max"
        ]].to_string(index=False))
    return 1 if invalid_imputed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
