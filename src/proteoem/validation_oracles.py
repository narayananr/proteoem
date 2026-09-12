"""Independent numerical oracles for validation benchmarks and tests.

This module is deliberately separate from :mod:`proteoem.em`.  It is not a
production fitting path.  The constrained optimizer uses Euclidean projection
and a projected-gradient line search, while the binary compatibility routine
implements the classical incidence-matrix update directly.  Neither routine
calls the ProteoEM EM kernel or its numerical helpers.

These oracles exist to check the core estimator by independent means.
:func:`maximize_fixed_mixture_on_simplex` maximizes the same fixed-component
mixture log-likelihood the EM fit ascends (manuscript main Eqs 5-9) but by a
different algorithm, so agreement confirms the EM fixed point is the constrained
maximum-likelihood solution (main Eq 9).  :func:`fit_binary_compatibility_em` is
the classical transcript-compatibility EM from RNA-seq quantification -- the same
hard-set formulation underlying the binary-incidence baseline of Supplementary
Methods S2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _positive_row_counts(counts: Any | None, n_rows: int) -> NDArray[np.float64]:
    if counts is None:
        return np.ones(n_rows, dtype=float)
    values = np.asarray(counts, dtype=float)
    if (
        values.ndim != 1
        or values.shape[0] != n_rows
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or values.sum() <= 0
    ):
        raise ValueError("counts must be finite, non-negative, and match the rows")
    return values


def _normalized_weights(weights: Any | None, n_candidates: int) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_candidates, 1.0 / n_candidates)
    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or values.shape[0] != n_candidates
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or values.sum() <= 0
    ):
        raise ValueError(
            "initial_weights must be finite, non-negative, and match the candidates"
        )
    return values / values.sum()


def _scaled_likelihoods(
    log_likelihoods: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    logs = np.asarray(log_likelihoods, dtype=float)
    if logs.ndim != 2 or min(logs.shape) < 1:
        raise ValueError("log_likelihoods must be a non-empty two-dimensional matrix")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_likelihoods may be finite or -inf, but not NaN or +inf")
    offsets = np.max(logs, axis=1)
    if np.any(~np.isfinite(offsets)):
        raise ValueError("every positive-count row must support at least one candidate")
    with np.errstate(under="ignore"):
        likelihoods = np.exp(logs - offsets[:, None])
    return likelihoods, offsets


def _project_probability_simplex(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Euclidean projection onto ``{x: x >= 0, sum(x) = 1}``.

    The sort-and-threshold construction is independent of the normalization
    used by the EM implementation and permits iterates on the simplex boundary.
    """

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1, dtype=float)
    active = ordered - cumulative / indices > 0
    rho = int(np.flatnonzero(active)[-1])
    threshold = cumulative[rho] / (rho + 1.0)
    projected = np.maximum(values - threshold, 0.0)
    projected /= projected.sum()
    return projected


def fixed_mixture_log_likelihood(
    log_likelihoods: Any,
    weights: Any,
    *,
    counts: Any | None = None,
) -> float:
    """Evaluate the fixed-component mixture objective without EM helpers."""

    likelihoods, offsets = _scaled_likelihoods(log_likelihoods)
    row_counts = _positive_row_counts(counts, likelihoods.shape[0])
    mixture = _normalized_weights(weights, likelihoods.shape[1])
    denominators = likelihoods @ mixture
    if np.any((row_counts > 0) & (denominators <= 0)):
        return -np.inf
    positive = row_counts > 0
    return float(
        np.dot(row_counts[positive], np.log(denominators[positive]) + offsets[positive])
    )


@dataclass(frozen=True)
class SimplexOptimizerResult:
    """Result from the benchmark-only projected-gradient optimizer."""

    weights: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    n_iter: int
    projected_gradient_residual: float
    line_search_failures: int


def maximize_fixed_mixture_on_simplex(
    log_likelihoods: Any,
    *,
    counts: Any | None = None,
    initial_weights: Any | None = None,
    max_iter: int = 50_000,
    tol: float = 2e-8,
    armijo: float = 1e-4,
    minimum_step: float = 1e-14,
) -> SimplexOptimizerResult:
    """Directly maximize a fixed-likelihood mixture on the simplex.

    This benchmark oracle uses projected gradient ascent with an Armijo
    backtracking line search on mean log likelihood.  It intentionally does
    not use latent assignments, expected counts, multiplicative updates, or
    any function from :mod:`proteoem.em`.
    """

    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(armijo) or not 0 < armijo < 1:
        raise ValueError("armijo must lie strictly between zero and one")
    if not np.isfinite(minimum_step) or minimum_step <= 0:
        raise ValueError("minimum_step must be finite and positive")

    likelihoods, offsets = _scaled_likelihoods(log_likelihoods)
    row_counts = _positive_row_counts(counts, likelihoods.shape[0])
    positive = row_counts > 0
    likelihoods = likelihoods[positive]
    offsets = offsets[positive]
    row_counts = row_counts[positive]
    total_count = float(row_counts.sum())
    weights = _normalized_weights(initial_weights, likelihoods.shape[1])

    def mean_centered_objective(candidate: NDArray[np.float64]) -> float:
        denominators = likelihoods @ candidate
        if np.any(denominators <= 0):
            return -np.inf
        return float(np.dot(row_counts, np.log(denominators)) / total_count)

    objective = mean_centered_objective(weights)
    step = 1.0
    converged = False
    residual = np.inf
    line_search_failures = 0
    n_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        denominators = likelihoods @ weights
        gradient = likelihoods.T @ (row_counts / denominators) / total_count
        unit_projection = _project_probability_simplex(weights + gradient)
        residual = float(np.max(np.abs(unit_projection - weights)))
        if residual <= tol:
            converged = True
            n_iter = iteration - 1
            break

        trial_step = step
        accepted = False
        while trial_step >= minimum_step:
            candidate = _project_probability_simplex(weights + trial_step * gradient)
            displacement = candidate - weights
            directional_gain = float(np.dot(gradient, displacement))
            candidate_objective = mean_centered_objective(candidate)
            if candidate_objective >= objective + armijo * directional_gain:
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            line_search_failures += 1
            n_iter = iteration - 1
            break

        weights = candidate
        objective = candidate_objective
        n_iter = iteration
        step = min(2.0, trial_step * 1.5)
    else:
        denominators = likelihoods @ weights
        gradient = likelihoods.T @ (row_counts / denominators) / total_count
        residual = float(
            np.max(
                np.abs(_project_probability_simplex(weights + gradient) - weights)
            )
        )

    absolute_objective = float(
        total_count * objective + np.dot(row_counts, offsets)
    )
    weights = np.asarray(weights, dtype=float)
    weights.setflags(write=False)
    return SimplexOptimizerResult(
        weights=weights,
        log_likelihood=absolute_objective,
        converged=converged,
        n_iter=n_iter,
        projected_gradient_residual=residual,
        line_search_failures=line_search_failures,
    )


@dataclass(frozen=True)
class BinaryCompatibilityResult:
    """Result from the independent binary incidence-matrix EM check."""

    weights: NDArray[np.float64]
    responsibilities: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    n_iter: int
    fixed_point_residual: float


def fit_binary_compatibility_em(
    compatibility: Any,
    *,
    counts: Any | None = None,
    initial_weights: Any | None = None,
    max_iter: int = 100_000,
    tol: float = 1e-12,
) -> BinaryCompatibilityResult:
    """Fit a transcript-style binary compatibility matrix independently.

    ``compatibility[i, k]`` states whether record/class ``i`` may originate
    from candidate ``k``.  The implementation works directly with incidence
    values and does not convert them to ProteoEM log likelihoods.
    """

    raw = np.asarray(compatibility)
    if raw.ndim != 2 or min(raw.shape) < 1 or not np.all(np.isin(raw, (0, 1))):
        raise ValueError("compatibility must be a non-empty binary matrix")
    incidence = np.asarray(raw, dtype=float)
    row_counts = _positive_row_counts(counts, incidence.shape[0])
    positive = row_counts > 0
    incidence = incidence[positive]
    row_counts = row_counts[positive]
    if np.any(incidence.sum(axis=1) == 0):
        raise ValueError("every positive-count row must support at least one candidate")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")

    total_count = float(row_counts.sum())
    weights = _normalized_weights(initial_weights, incidence.shape[1])
    converged = False
    n_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        denominators = incidence @ weights
        if np.any(denominators <= 0):
            raise ValueError("current weights leave a compatibility row unsupported")
        responsibilities = incidence * weights[None, :] / denominators[:, None]
        updated = np.sum(row_counts[:, None] * responsibilities, axis=0) / total_count
        change = float(np.max(np.abs(updated - weights)))
        weights = updated
        n_iter = iteration
        if change <= tol:
            converged = True
            break

    denominators = incidence @ weights
    responsibilities = incidence * weights[None, :] / denominators[:, None]
    fixed_point = (
        np.sum(row_counts[:, None] * responsibilities, axis=0) / total_count
    )
    residual = float(np.max(np.abs(fixed_point - weights)))
    objective = float(np.dot(row_counts, np.log(denominators)))
    for array in (weights, responsibilities):
        array.setflags(write=False)
    return BinaryCompatibilityResult(
        weights=weights,
        responsibilities=responsibilities,
        log_likelihood=objective,
        converged=converged,
        n_iter=n_iter,
        fixed_point_residual=residual,
    )


def deterministic_profile_compatibility(
    observations: Any, profiles: Any
) -> NDArray[np.bool_]:
    """Construct wildcard compatibility without affinity-likelihood code."""

    observed = np.asarray(observations)
    reference = np.asarray(profiles)
    if observed.ndim != 2 or min(observed.shape) < 1:
        raise ValueError("observations must be a non-empty two-dimensional matrix")
    if reference.ndim != 2 or min(reference.shape) < 1:
        raise ValueError("profiles must be a non-empty two-dimensional matrix")
    if observed.shape[1] != reference.shape[1]:
        raise ValueError("observations and profiles must contain the same columns")
    if not np.all(np.isin(observed, (-1, 0, 1))):
        raise ValueError("observations must contain only -1, 0, and 1")
    if not np.all(np.isin(reference, (0, 1))):
        raise ValueError("profiles must be binary")
    called = observed != -1
    matches = (~called[:, None, :]) | (
        observed[:, None, :] == reference[None, :, :]
    )
    compatibility = np.all(matches, axis=2)
    if np.any(~np.any(compatibility, axis=1)):
        raise ValueError("some observations have no deterministic compatible profile")
    return compatibility


def likelihood_ratio_compatibility(
    log_likelihoods: Any, minimum_relative_likelihood: float
) -> NDArray[np.bool_]:
    """Threshold graded evidence into row-relative binary compatibility.

    A candidate is retained when its likelihood is at least ``threshold``
    times the maximum likelihood in that row.  At a zero threshold every
    positive-likelihood candidate is retained; structural zeros remain out.
    """

    logs = np.asarray(log_likelihoods, dtype=float)
    if logs.ndim != 2 or min(logs.shape) < 1:
        raise ValueError("log_likelihoods must be a non-empty two-dimensional matrix")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_likelihoods may be finite or -inf, but not NaN or +inf")
    maxima = np.max(logs, axis=1)
    if np.any(~np.isfinite(maxima)):
        raise ValueError("every row must have a positive-likelihood candidate")
    threshold = float(minimum_relative_likelihood)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("minimum_relative_likelihood must lie in [0, 1]")
    if threshold == 0:
        return np.isfinite(logs)
    log_threshold = np.log(threshold)
    return np.isfinite(logs) & (logs - maxima[:, None] >= log_threshold)
