"""Alignment-profile adapters and simple decoding baselines.

The two deliberately reduced comparators of Supplementary Methods S2.5, which the
benchmark contrasts against weighted EM. ``alignment_profile_log_likelihoods``
rounds the calibrated likelihoods to hard 0/1 compatibility (the input to
binary-profile EM); ``hard_assignment_counts`` assigns each molecule to its single
maximum-likelihood origin (top-likelihood counting). Both reduce each trace to a
hard yes/no rather than keeping the graded likelihood, and both lose accuracy
relative to weighted EM (manuscript Section 3.2).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .data import AggregatedTraces, as_aggregated
from .emissions import validate_profiles


def alignment_profile_log_likelihoods(
    observations: Any | AggregatedTraces,
    profiles: Any,
    *,
    mode: Literal["best", "exact"] = "best",
    max_mismatches: int = 0,
) -> NDArray[np.float64]:
    """Convert binary probe traces to a 0/1 candidate-alignment profile
    (the binary-profile baseline of Supplementary Methods S2.5).

    Returned values are log incidence: zero for compatible candidates and
    negative infinity otherwise.  Missing calls are wildcards.  ``best``
    retains every candidate in the minimum-mismatch stratum, analogous to an
    aligner's best stratum.  ``exact`` retains candidates with at most
    ``max_mismatches`` and errors if a row has none.
    """

    if mode not in {"best", "exact"}:
        raise ValueError("mode must be 'best' or 'exact'")
    if (
        not isinstance(max_mismatches, (int, np.integer))
        or isinstance(max_mismatches, bool)
        or max_mismatches < 0
    ):
        raise ValueError("max_mismatches must be a non-negative integer")

    data = as_aggregated(observations)
    candidate_profiles = validate_profiles(profiles)
    if candidate_profiles.shape[1] != data.n_probes:
        raise ValueError("profiles and observations must contain the same probes")

    mismatches = np.zeros(
        (data.n_unique, candidate_profiles.shape[0]), dtype=np.int32
    )
    for probe in range(data.n_probes):
        observed = data.mask[:, probe]
        if np.any(observed):
            mismatches[observed] += (
                data.traces[observed, probe, None]
                != candidate_profiles[None, :, probe]
            )

    if mode == "best":
        compatible = mismatches == np.min(mismatches, axis=1, keepdims=True)
    else:
        compatible = mismatches <= max_mismatches
        if np.any(~np.any(compatible, axis=1)):
            rows = np.flatnonzero(~np.any(compatible, axis=1)).tolist()
            raise ValueError(
                "some traces have no compatible candidate at the requested "
                f"mismatch threshold (aggregated rows {rows})"
            )

    return np.where(compatible, 0.0, -np.inf)


def hard_assignment_counts(
    log_likelihoods: Any,
    *,
    counts: Any | None = None,
    split_ties: bool = True,
) -> NDArray[np.float64]:
    """Count maximum-likelihood candidate calls, optionally splitting ties
    (top-likelihood counting, Supplementary Methods S2.5)."""

    logs = np.asarray(log_likelihoods, dtype=float)
    if logs.ndim != 2 or min(logs.shape) < 1:
        raise ValueError("log_likelihoods must be a non-empty two-dimensional matrix")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_likelihoods may be finite or -inf, but not NaN or +inf")
    if np.any(~np.any(np.isfinite(logs), axis=1)):
        raise ValueError("every row must support at least one candidate")
    row_counts = (
        np.ones(logs.shape[0], dtype=float)
        if counts is None
        else np.asarray(counts, dtype=float)
    )
    if (
        row_counts.ndim != 1
        or row_counts.shape[0] != logs.shape[0]
        or not np.all(np.isfinite(row_counts))
        or np.any(row_counts < 0)
    ):
        raise ValueError("counts must be finite, non-negative, and match the rows")

    maxima = np.max(logs, axis=1, keepdims=True)
    winners = logs == maxima
    if split_ties:
        allocations = winners / winners.sum(axis=1, keepdims=True)
        return np.sum(row_counts[:, None] * allocations, axis=0)
    selected = np.argmax(logs, axis=1)
    return np.bincount(
        selected, weights=row_counts, minlength=logs.shape[1]
    ).astype(float)

