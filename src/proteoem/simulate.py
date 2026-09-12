"""Simulation from the fixed-emission generative model.

Draws molecule identities from a mixture over origins and then, given each
identity and the fixed emission matrix ``Q`` (manuscript main Eq 1), draws
independent per-cycle binary calls (main Eq 2). This is the generative model of
main Eq 12, written at the level of *analyzed* traces: ``weights`` is the
composition among the traces produced, so it corresponds to the mixture ``pi``
the EM fit estimates, not the upstream source composition ``theta``. For a
simulation that starts from source molecules and records recovery and a trace
gate separately (main Section 2.3), use :mod:`proteoem.selection_simulate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import AggregatedTraces, aggregate_traces
from .emissions import build_emission_matrix


@dataclass(frozen=True)
class SimulationResult:
    observations: NDArray[np.int8]
    identities: NDArray[np.int64]
    weights: NDArray[np.float64]
    Q: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("observations", "identities", "weights", "Q"):
            array = np.asarray(getattr(self, name)).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def aggregated(self) -> AggregatedTraces:
        return aggregate_traces(self.observations)

    @property
    def analyzed_composition(self) -> NDArray[np.float64]:
        """Alias clarifying that ``weights`` generate the returned traces."""

        return self.weights


def _normalize_weights(weights: Any, n_candidates: int) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_candidates, 1.0 / n_candidates)
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.shape[0] != n_candidates:
        raise ValueError(f"weights must contain one value for {n_candidates} candidates")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or array.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    return array / array.sum()


def _missing_probabilities(missing_rate: Any, n_probes: int) -> NDArray[np.float64]:
    try:
        rates = np.broadcast_to(np.asarray(missing_rate, dtype=float), (n_probes,))
    except (TypeError, ValueError) as exc:
        raise ValueError("missing_rate must be scalar or contain one value per probe") from exc
    if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError("missing_rate must contain probabilities in [0, 1]")
    return np.asarray(rates, dtype=float)


def simulate_traces(
    profiles: Any,
    n_molecules: int,
    *,
    weights: Any | None = None,
    alpha: Any = 0.95,
    beta: Any = 0.05,
    Q: Any | None = None,
    missing_rate: Any = 0.0,
    seed: int | np.integer | None = None,
) -> SimulationResult:
    """Simulate identities and ``-1/0/1`` affinity traces.

    All randomness comes from a local ``numpy.random.Generator`` initialized by
    ``seed``; global NumPy random state is neither read nor modified.
    """

    if not isinstance(n_molecules, (int, np.integer)) or isinstance(n_molecules, bool):
        raise ValueError("n_molecules must be a positive integer")
    if n_molecules <= 0:
        raise ValueError("n_molecules must be a positive integer")
    q = build_emission_matrix(profiles, alpha=alpha, beta=beta, Q=Q)
    mixture = _normalize_weights(weights, q.shape[0])
    missing = _missing_probabilities(missing_rate, q.shape[1])

    rng = np.random.default_rng(seed)
    # Identity ~ mixture, then each cycle is an independent Bernoulli(q) draw
    # under the row of Q for that origin (main Eq 1-2): the generative model of
    # main Eq 12.
    identities = rng.choice(q.shape[0], size=int(n_molecules), p=mixture)
    positive = rng.random((int(n_molecules), q.shape[1])) < q[identities]
    observations = positive.astype(np.int8)
    if np.any(missing > 0):
        missing_mask = rng.random(observations.shape) < missing[None, :]
        observations[missing_mask] = -1

    return SimulationResult(
        observations=observations,
        identities=np.asarray(identities, dtype=np.int64),
        weights=mixture,
        Q=q,
    )
