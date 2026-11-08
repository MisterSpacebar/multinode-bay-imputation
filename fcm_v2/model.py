"""Direct-horizon physical concept maps, fitted without future information.

This is a learned *lagged association* map, not a causal mechanism. W[i, j]
connects issue-day concept i to concept j at t+h. All normalization, output
transforms and intercepts are fitted strictly on earlier training pairs.

Two activation functions are available:
- linear: penalized least squares in standardized concept space
- sigmoid: penalized least squares on the logit of the bounded output,
  with exactly the fitted sigmoid used at prediction time

The sigmoid version is a bounded, nonlinear FCM variant. It deliberately
does NOT rescale its fitted weights to [-1,1], which would alter its model.
No future exogenous drivers are clamped in or supplied at forecast time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DO_CONCEPTS = ("hours_lt4", "odo_mean")
PHYSICAL_CONCEPTS = DO_CONCEPTS + ("temp_mean", "sal_mean")
BOUNDS = {
    "hours_lt4": (0.0, 24.0),
    "odo_mean": (0.0, 20.0),
    "temp_mean": (-5.0, 45.0),
    "sal_mean": (0.0, 45.0),
}
EPS = 1e-4


@dataclass
class ConceptMap:
    concepts: tuple[str, ...]
    activation: str
    horizon_days: int
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    n_target_pairs: np.ndarray

    def predict(self, current) -> np.ndarray:
        state = np.asarray(current, dtype=float)
        if state.shape != (len(self.concepts),) or not np.isfinite(state).all():
            raise ValueError("Current concept state must be complete and finite")
        z = (state - self.center) / self.scale
        output = z @ self.weights + self.intercept
        if self.activation == "sigmoid":
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -50, 50)))
            values = np.array([
                BOUNDS[c][0] + output[i] * (BOUNDS[c][1] - BOUNDS[c][0])
                for i, c in enumerate(self.concepts)
            ], dtype=float)
        elif self.activation == "linear":
            values = output * self.scale + self.center
        else:
            raise ValueError(f"Unknown activation {self.activation}")
        # Sanity clipping, using declared physical/definition bounds only.
        for i, concept in enumerate(self.concepts):
            if np.isfinite(values[i]):
                values[i] = np.clip(values[i], *BOUNDS[concept])
        return values


def prepare_targets(daily: pd.DataFrame, concepts: tuple[str, ...],
                    horizon: int) -> pd.DataFrame:
    """Append exact-future-date concept targets; leave gaps unfilled."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = daily.copy()
    for concept in concepts:
        if concept not in frame:
            frame[concept] = np.nan
        frame[concept] = pd.to_numeric(frame[concept], errors="coerce")
        frame[f"target_{concept}"] = frame[concept].shift(-horizon)
    frame["target_date"] = frame.index + pd.Timedelta(days=horizon)
    return frame


def training_pair_count(frame: pd.DataFrame, concepts: tuple[str, ...],
                        issued_at: pd.Timestamp) -> int:
    valid_inputs = frame[list(concepts)].notna().all(axis=1)
    eligible = (valid_inputs & frame["target_hours_lt4"].notna()
                & (frame["target_date"] < issued_at))
    return int(eligible.sum())


def fit_map(frame: pd.DataFrame, concepts: tuple[str, ...],
            horizon: int, issued_at: pd.Timestamp,
            activation: str = "linear", alpha: float = 10.0,
            min_train: int = 40) -> ConceptMap | None:
    """Fit a lagged concept transition as of the END of issued_at.

    Each destination concept can have a different number of observed targets.
    The risk/hypoxia target (hours_lt4) must have >=min_train pairs. Fewer
    physical follow-up measurements do not invalidate an otherwise sound
    DO target fit; untrained destinations remain NaN in the map.
    """
    if activation not in ("linear", "sigmoid"):
        raise ValueError(f"Unsupported activation: {activation}")
    if alpha <= 0 or min_train < 2:
        raise ValueError("alpha must be positive; min_train >= 2")
    concepts = tuple(concepts)
    if not concepts or concepts[0] != "hours_lt4":
        raise ValueError("The first concept must be hours_lt4")
    valid_inputs = frame[list(concepts)].notna().all(axis=1)
    before_issue = frame["target_date"] < issued_at
    do_pairs = valid_inputs & before_issue & frame["target_hours_lt4"].notna()
    if int(do_pairs.sum()) < min_train:
        return None

    x_fit = frame.loc[do_pairs, list(concepts)].to_numpy(dtype=float)
    center = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-8] = 1.0
    n = len(concepts)
    W = np.full((n, n), np.nan, dtype=float)
    intercept = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=int)
    for j, target in enumerate(concepts):
        observed_pairs = valid_inputs & before_issue & frame[f"target_{target}"].notna()
        count = int(observed_pairs.sum())
        counts[j] = count
        required = min_train if j == 0 else max(10, min_train // 2)
        if count < required:
            continue
        x = frame.loc[observed_pairs, list(concepts)].to_numpy(dtype=float)
        y = frame.loc[observed_pairs, f"target_{target}"].to_numpy(dtype=float)
        z = (x - center) / scale
        if activation == "linear":
            yt = (y - center[j]) / scale[j]
        else:
            low, high = BOUNDS[target]
            ratio = np.clip((y - low) / (high - low), EPS, 1.0 - EPS)
            yt = np.log(ratio / (1.0 - ratio))
        xm, ym = z.mean(axis=0), float(yt.mean())
        zc = z - xm
        w = np.linalg.solve(
            zc.T @ zc + alpha * np.eye(n),
            zc.T @ (yt - ym),
        )
        W[:, j] = w
        intercept[j] = ym - float(xm @ w)
    return ConceptMap(concepts, activation, horizon, center, scale,
                      W, intercept, counts)


def edge_rows(model: ConceptMap, node: str, issue_date: pd.Timestamp) -> list[dict]:
    """Coefficients are signed, standardized *associations*, not causes."""
    rows = []
    for i, source in enumerate(model.concepts):
        for j, target in enumerate(model.concepts):
            w = float(model.weights[i, j])
            rows.append({
                "node": node, "asof_date": issue_date.date().isoformat(),
                "horizon_days": model.horizon_days,
                "activation": model.activation,
                "concept_set": ("physical" if len(model.concepts) > 2 else "do"),
                "source": source, "target": target,
                "weight": w, "intercept": float(model.intercept[j]),
                "n_train_target": int(model.n_target_pairs[j]),
                "weight_units": ("standardized_outcome_per_1sd_input"
                                 if model.activation == "linear"
                                 else "logit_outcome_per_1sd_input"),
                "interpretation": "lagged_association_not_causal",
            })
    return rows


def sensitivity_rows(model: ConceptMap, node: str,
                     issued_at: pd.Timestamp, current) -> list[dict]:
    """Single-input perturbations; not interventions or causal predictions."""
    if tuple(model.concepts) != PHYSICAL_CONCEPTS:
        return []
    state = np.asarray(current, dtype=float)
    original = float(model.predict(state)[0])
    rows = []
    for concept, delta in (("temp_mean", 1.0), ("sal_mean", 1.0)):
        changed = state.copy()
        j = model.concepts.index(concept)
        changed[j] = np.clip(changed[j] + delta, *BOUNDS[concept])
        if changed[j] == state[j]:
            continue
        altered = float(model.predict(changed)[0])
        rows.append({
            "node": node, "asof_date": issued_at.date().isoformat(),
            "horizon_days": model.horizon_days,
            "activation": model.activation,
            "perturbed_concept": concept,
            "requested_delta": delta,
            "actual_delta": float(changed[j] - state[j]),
            "base_predicted_hours_lt4": original,
            "altered_predicted_hours_lt4": altered,
            "predicted_difference_hours": altered - original,
            "interpretation": "model_input_sensitivity_not_causal",
        })
    return rows
