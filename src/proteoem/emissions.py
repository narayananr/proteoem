"""Emission models for binary iterative-affinity traces.

Builds the fixed emission matrix ``Q`` (manuscript main Eq 1,
``q_kj = E_kj * alpha_j + (1 - E_kj) * beta_j``) and turns a trace of calls into a
per-origin log-likelihood row. Given the origin and ``Q``, the cycle calls are
conditionally independent (main Eq 2), so a trace's likelihood is the product of
per-cycle factors: ``q`` for a positive call, ``1 - q`` for a negative, and a
neutral factor of 1 for a missing (NA) call under ignorable missingness (main
Eq 4). ``Q`` is calibrated beforehand and held fixed during the abundance fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import (
    AggregatedTraces,
    ProbeCountClasses,
    as_aggregated,
    validate_cycle_to_probe,
)


def validate_profiles(profiles: Any) -> NDArray[np.int8]:
    """Validate a candidate-by-probe binary profile matrix."""

    array = np.asarray(profiles)
    if array.ndim != 2:
        raise ValueError("profiles must be a candidate-by-probe matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("profiles must contain at least one candidate and probe")
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError("candidate profiles must be binary")
    return np.asarray(array, dtype=np.int8)


def _broadcast_probability(value: Any, shape: tuple[int, int], name: str) -> NDArray:
    try:
        array = np.asarray(value, dtype=float)
        broadcast = np.broadcast_to(array, shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to candidate-by-probe shape"
        ) from exc
    if not np.all(np.isfinite(broadcast)) or np.any((broadcast < 0) | (broadcast > 1)):
        raise ValueError(f"{name} probabilities must be finite and in [0, 1]")
    return np.asarray(broadcast, dtype=float)


def build_emission_matrix(
    profiles: Any | None,
    *,
    alpha: Any = 0.95,
    beta: Any = 0.05,
    Q: Any | None = None,
) -> NDArray[np.float64]:
    """Return ``P(binding=1 | candidate, probe)``, the fixed emission matrix ``Q``
    (the structured model of main Eq 1).

    ``alpha`` is the on-target positive-call probability (profile value 1) and
    ``beta`` the off-target positive-call probability (profile value 0); each may
    be a scalar, a per-probe vector, or an array broadcastable to the profile
    shape.  A supplied full ``Q`` takes precedence and is returned as a copy.

    Examples
    --------
    >>> import numpy as np
    >>> profiles = np.array([[1, 0, 1],    # candidate 0 carries probes 0 and 2
    ...                      [0, 1, 1]])    # candidate 1 carries probes 1 and 2
    >>> build_emission_matrix(profiles, alpha=0.9, beta=0.1)
    array([[0.9, 0.1, 0.9],
           [0.1, 0.9, 0.9]])
    """

    if Q is not None:
        q = np.asarray(Q, dtype=float)
        if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
            raise ValueError("Q must be a non-empty candidate-by-probe matrix")
        if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
            raise ValueError("Q probabilities must be finite and in [0, 1]")
        if profiles is not None:
            profile_array = validate_profiles(profiles)
            if q.shape != profile_array.shape:
                raise ValueError(f"Q has shape {q.shape}; expected {profile_array.shape}")
        return q.copy()

    if profiles is None:
        raise ValueError("profiles are required when Q is not supplied")
    profile_array = validate_profiles(profiles)
    shape = profile_array.shape

    sensitivity = _broadcast_probability(alpha, shape, "alpha")
    false_positive = _broadcast_probability(beta, shape, "beta")
    return np.where(profile_array == 1, sensitivity, false_positive)


def log_likelihood_from_aggregated(
    data: AggregatedTraces, Q: Any
) -> NDArray[np.float64]:
    """Candidate log likelihoods for unique trace/mask patterns (main Eq 2, with
    missing calls marginalized per main Eq 4)."""

    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[1] != data.n_probes:
        raise ValueError("Q must be candidate-by-probe and match the observations")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q probabilities must be finite and in [0, 1]")

    with np.errstate(divide="ignore"):
        log_positive = np.log(q)
        log_negative = np.log1p(-q)

    n_traces = data.n_unique
    n_candidates = q.shape[0]
    result = np.zeros((n_traces, n_candidates), dtype=float)
    for probe in range(data.n_probes):
        observed = data.mask[:, probe]
        if not np.any(observed):
            continue
        values = data.traces[observed, probe]
        contribution = np.where(
            values[:, None] == 1,
            log_positive[None, :, probe],
            log_negative[None, :, probe],
        )
        result[observed] += contribution
    return result


def log_likelihood_from_probe_counts(
    data: ProbeCountClasses, Q: Any
) -> NDArray[np.float64]:
    """Compute *relative* candidate log likelihoods for probe-count classes
    (the schedule-aware trace-likelihood classes of main Eq 11 / Supplementary
    S2.4). Repeated applications of one calibrated probe reduce to per-probe
    sufficient counts -- how many positive and how many negative -- so a row is a
    count-weighted sum of ``log q`` and ``log(1 - q)`` over the retained probes.

    ``data.log_likelihood_offset`` holds the origin-independent factors removed
    during class construction (e.g. the pan-tau probes, identical across every
    origin). They are constant in the mixture weights and so are left out of these
    assignment-relevant rows, but kept for the absolute log-likelihood.
    """

    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[1] != data.n_logical_probes:
        raise ValueError("Q must be candidate-by-logical-probe")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q probabilities must be finite and in [0, 1]")

    selected_q = q[:, data.probe_indices]
    with np.errstate(divide="ignore"):
        log_positive = np.log(selected_q)
        log_negative = np.log1p(-selected_q)

    result = np.zeros((data.n_unique, q.shape[0]), dtype=float)
    for column in range(data.n_retained_probes):
        positives = data.positive_counts[:, column]
        negatives = data.observed_counts[:, column] - positives
        positive_rows = positives > 0
        negative_rows = negatives > 0
        if np.any(positive_rows):
            result[positive_rows] += (
                positives[positive_rows, None] * log_positive[None, :, column]
            )
        if np.any(negative_rows):
            result[negative_rows] += (
                negatives[negative_rows, None] * log_negative[None, :, column]
            )
    return result


def candidate_log_likelihoods(
    observations: Any,
    Q: Any,
    *,
    cycle_to_probe: Any | None = None,
) -> NDArray[np.float64]:
    """Return one candidate log-likelihood row per input observation.

    Given the origin and ``Q`` the calls are conditionally independent (main
    Eq 2), so each observed call contributes ``log q`` for a positive (``1``) and
    ``log(1 - q)`` for a negative (``0``).  Missing cycles (``-1``) marginalize to
    a factor of 1 under ignorable missingness (main Eq 4) and add exactly zero, so
    they never act as negatives.  Pass ``cycle_to_probe`` to map physical cycles onto the
    logical-probe columns of ``Q`` (repeated indices = repeated applications of
    one calibrated probe).

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([[0.9, 0.1],           # candidate 0
    ...               [0.1, 0.9]])          # candidate 1
    >>> obs = np.array([[1, -1]])           # probe 0 positive, probe 1 missing
    >>> candidate_log_likelihoods(obs, Q).round(4)   # the missing call adds 0
    array([[-0.1054, -2.3026]])
    >>> # three physical cycles applying logical probes [0, 0, 1]
    >>> Qs = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> candidate_log_likelihoods([[1, 0, 1]], Qs, cycle_to_probe=[0, 0, 1]).round(4)
    array([[-4.7105, -2.0557]])
    """

    if isinstance(observations, ProbeCountClasses):
        if cycle_to_probe is not None:
            raise ValueError("cycle_to_probe must not be supplied with ProbeCountClasses")
        return log_likelihood_from_probe_counts(observations, Q)

    data = as_aggregated(observations)
    q = np.asarray(Q, dtype=float)
    if cycle_to_probe is not None:
        if q.ndim != 2:
            raise ValueError("Q must be candidate-by-logical-probe")
        mapping = validate_cycle_to_probe(
            cycle_to_probe,
            n_cycles=data.n_probes,
            n_logical_probes=q.shape[1],
        )
        q = q[:, mapping]
    unique = log_likelihood_from_aggregated(data, q)
    if isinstance(observations, AggregatedTraces):
        return unique

    from .data import validate_observations

    # ``as_aggregated`` sorts unique patterns, so expand through a canonical
    # lookup to preserve the caller-visible row order.
    raw = validate_observations(observations, n_probes=data.n_probes)
    key_to_row = {
        tuple(row.tolist()): likelihood
        for row, likelihood in zip(data.as_observations(), unique, strict=True)
    }
    return np.vstack([key_to_row[tuple(row.tolist())] for row in raw])
