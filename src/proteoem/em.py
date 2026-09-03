"""Fixed-emission finite-mixture expectation maximization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import (
    AggregatedTraces,
    ProbeCountClasses,
    aggregate_probe_counts,
    as_aggregated,
    compress_probe_count_classes,
    validate_cycle_to_probe,
)
from .emissions import (
    build_emission_matrix,
    log_likelihood_from_aggregated,
    log_likelihood_from_probe_counts,
)
from .equivalence import find_observable_groups


DiagnosticValue = float | int | bool | str
AggregatedEvidence = AggregatedTraces | ProbeCountClasses


def logsumexp(values: Any, axis: int | None = None) -> NDArray | np.float64:
    """Stable ``log(sum(exp(values)))`` supporting all-negative-infinity rows."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("logsumexp is undefined for an empty array")
    maximum = np.max(array, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid="ignore", under="ignore", divide="ignore"):
        shifted = np.where(finite, array - maximum, -np.inf)
        total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        result = np.where(finite, maximum + np.log(total), maximum)
    if axis is None:
        return np.float64(result.squeeze())
    return np.squeeze(result, axis=axis)


def _normalize_weights(weights: Any, n_candidates: int) -> NDArray[np.float64]:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.shape[0] != n_candidates:
        raise ValueError(f"weights must contain one value for {n_candidates} candidates")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or array.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    return np.asarray(array / array.sum(), dtype=float)


def _posterior_from_logs(
    component_logs: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized posteriors and row log normalizers.

    Rows are centered before adding log mixture weights. This preserves the
    row-scale invariance of the likelihood interface and prevents a large
    origin-independent offset from numerically erasing the mixture weights.
    """

    row_offsets = np.max(component_logs, axis=1)
    impossible = ~np.isfinite(row_offsets)
    if np.any(impossible):
        indices = np.flatnonzero(impossible).tolist()
        raise ValueError(
            "some trace patterns have zero probability under all candidates "
            f"with positive weight (aggregated rows {indices})"
        )
    centered = component_logs - row_offsets[:, None]
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    joint = centered + log_weights[None, :]
    centered_normalizers = np.asarray(logsumexp(joint, axis=1))
    impossible = ~np.isfinite(centered_normalizers)
    if np.any(impossible):
        indices = np.flatnonzero(impossible).tolist()
        raise ValueError(
            "some trace patterns have zero probability under all candidates "
            f"with positive weight (aggregated rows {indices})"
        )
    joint -= centered_normalizers[:, None]
    np.exp(joint, out=joint)
    joint /= joint.sum(axis=1)[:, None]
    return joint, centered_normalizers + row_offsets


def _validate_block_size(block_size: int | None, n_rows: int) -> int:
    if block_size is None:
        return n_rows
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer or None")
    return min(int(block_size), n_rows)


def _e_step_statistics(
    component_logs: NDArray[np.float64],
    multiplicities: NDArray[np.int64],
    weights: NDArray[np.float64],
    *,
    block_size: int,
    return_responsibilities: bool,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None]:
    """Accumulate one E-step without multiple full posterior temporaries."""

    n_rows, n_candidates = component_logs.shape
    expected = np.zeros(n_candidates, dtype=float)
    log_likelihood = 0.0
    responsibilities = (
        np.empty((n_rows, n_candidates), dtype=float)
        if return_responsibilities
        else None
    )
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        posterior, normalizers = _posterior_from_logs(
            component_logs[start:stop], weights
        )
        block_counts = multiplicities[start:stop]
        expected += np.sum(block_counts[:, None] * posterior, axis=0)
        log_likelihood += float(np.dot(block_counts, normalizers))
        if responsibilities is not None:
            responsibilities[start:stop] = posterior
    return expected, log_likelihood, responsibilities


@dataclass(frozen=True)
class _KernelResult:
    weights: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    responsibilities: NDArray[np.float64] | None
    log_likelihood_history: NDArray[np.float64]
    centered_log_likelihood_history: NDArray[np.float64]
    converged: bool
    n_iter: int
    diagnostics: dict[str, DiagnosticValue]


def _fit_component_logs(
    log_likelihoods: Any,
    *,
    counts: Any | None,
    initial_weights: Any | None,
    max_iter: int,
    tol: float,
    log_likelihood_offset: float = 0.0,
    block_size: int | None = 4096,
    return_responsibilities: bool = True,
    copy_logs: bool = True,
) -> _KernelResult:
    """Shared, row-centered EM kernel used by both public fitters."""

    raw_logs = np.asarray(log_likelihoods, dtype=float)
    if raw_logs.ndim != 2 or min(raw_logs.shape) < 1:
        raise ValueError("log_likelihoods must be a non-empty two-dimensional matrix")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if not isinstance(return_responsibilities, (bool, np.bool_)):
        raise ValueError("return_responsibilities must be boolean")
    if not np.isfinite(log_likelihood_offset):
        raise ValueError("log_likelihood_offset must be finite")

    n_input_rows, n_candidates = raw_logs.shape
    if counts is None:
        multiplicities = np.ones(n_input_rows, dtype=np.int64)
    else:
        raw_counts = np.asarray(counts)
        if raw_counts.ndim != 1 or raw_counts.shape[0] != n_input_rows:
            raise ValueError("counts must contain one value per likelihood row")
        try:
            numeric_counts = np.asarray(raw_counts, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "counts must be finite non-negative integers with positive sum"
            ) from exc
        if (
            not np.all(np.isfinite(numeric_counts))
            or np.any(numeric_counts < 0)
            or not np.all(numeric_counts == np.floor(numeric_counts))
            or numeric_counts.sum() <= 0
        ):
            raise ValueError(
                "counts must be finite non-negative integers with positive sum"
            )
        multiplicities = numeric_counts.astype(np.int64)

    positive = multiplicities > 0
    selected_logs = raw_logs if np.all(positive) else raw_logs[positive]
    component_logs = np.array(selected_logs, dtype=float, copy=copy_logs)
    multiplicities = multiplicities[positive]
    if np.any(np.isnan(component_logs)) or np.any(np.isposinf(component_logs)):
        raise ValueError("log_likelihoods may be finite or -inf, but not NaN or +inf")
    if np.any(~np.any(np.isfinite(component_logs), axis=1)):
        raise ValueError("every positive-count likelihood row must support a candidate")

    row_offsets = np.max(component_logs, axis=1)
    component_logs -= row_offsets[:, None]
    total_offset = float(log_likelihood_offset + np.dot(multiplicities, row_offsets))
    n_rows = component_logs.shape[0]
    block = _validate_block_size(block_size, n_rows)
    n_observations = int(multiplicities.sum())
    weights = (
        np.full(n_candidates, 1.0 / n_candidates)
        if initial_weights is None
        else _normalize_weights(initial_weights, n_candidates)
    )

    support_entries = 0
    uninformative_rows = np.empty(n_rows, dtype=bool)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        log_block = component_logs[start:stop]
        finite_block = np.isfinite(log_block)
        support_entries += int(np.count_nonzero(finite_block))
        uninformative_rows[start:stop] = np.all(finite_block, axis=1) & np.all(
            log_block == log_block[:, [0]], axis=1
        )
    informative_rows = ~uninformative_rows
    if np.all(informative_rows):
        informative_logs = component_logs
        informative_counts = multiplicities
    else:
        informative_logs = component_logs[informative_rows]
        informative_counts = multiplicities[informative_rows]
    n_informative_observations = int(informative_counts.sum())
    n_uninformative_observations = n_observations - n_informative_observations

    centered_history: list[float]
    converged = False
    n_iter = 0
    max_weight_change = 0.0
    last_gain = 0.0

    if n_informative_observations == 0:
        centered_history = [0.0]
        informative_expected = np.zeros(n_candidates, dtype=float)
        converged = True
    else:
        informative_block = _validate_block_size(block_size, informative_logs.shape[0])
        informative_expected, centered_log_likelihood, _ = _e_step_statistics(
            informative_logs,
            informative_counts,
            weights,
            block_size=informative_block,
            return_responsibilities=False,
        )
        centered_history = [centered_log_likelihood]
        for iteration in range(1, int(max_iter) + 1):
            updated = _normalize_weights(informative_expected, n_candidates)
            updated_expected, updated_log_likelihood, _ = _e_step_statistics(
                informative_logs,
                informative_counts,
                updated,
                block_size=informative_block,
                return_responsibilities=False,
            )
            last_gain = updated_log_likelihood - centered_log_likelihood
            max_weight_change = float(np.max(np.abs(updated - weights)))
            weights = updated
            informative_expected = updated_expected
            centered_log_likelihood = updated_log_likelihood
            centered_history.append(centered_log_likelihood)
            n_iter = iteration
            likelihood_stable = abs(last_gain) <= tol * (
                1.0 + abs(centered_log_likelihood)
            )
            weights_stable = max_weight_change <= max(tol, np.finfo(float).eps)
            if likelihood_stable and weights_stable:
                converged = True
                break

    expected_counts = informative_expected + n_uninformative_observations * weights
    responsibilities: NDArray[np.float64] | None = None
    if return_responsibilities:
        expected_counts, _, responsibilities = _e_step_statistics(
            component_logs,
            multiplicities,
            weights,
            block_size=block,
            return_responsibilities=True,
        )

    centered_history_array = np.asarray(centered_history, dtype=float)
    history_array = centered_history_array + total_offset
    gains = np.diff(centered_history_array)
    monotonic_tolerance = 1e-10 * (1.0 + np.abs(centered_history_array[:-1]))
    monotonic = bool(np.all(gains >= -monotonic_tolerance))
    minimum_gain = float(np.min(gains)) if gains.size else 0.0
    weight_sum_error = float(abs(weights.sum() - 1.0))
    expected_count_total_error = float(abs(expected_counts.sum() - n_observations))
    if n_informative_observations:
        fixed_point = informative_expected / n_informative_observations
        terminal_em_residual = float(np.max(np.abs(fixed_point - weights)))
    else:
        terminal_em_residual = 0.0
    posterior_sum_error = (
        float(np.max(np.abs(responsibilities.sum(axis=1) - 1.0)))
        if responsibilities is not None
        else 0.0
    )
    invariant_tolerance = 1e-10 * max(1, n_observations)
    if weight_sum_error > 1e-12 or expected_count_total_error > invariant_tolerance:
        raise FloatingPointError("EM normalization invariant failed")

    for array in (
        weights,
        expected_counts,
        history_array,
        centered_history_array,
    ):
        array.setflags(write=False)
    if responsibilities is not None:
        responsibilities.setflags(write=False)

    diagnostics: dict[str, DiagnosticValue] = {
        "initial_log_likelihood": float(history_array[0]),
        "final_log_likelihood": float(history_array[-1]),
        "initial_centered_log_likelihood": float(centered_history_array[0]),
        "final_centered_log_likelihood": float(centered_history_array[-1]),
        "log_likelihood_offset": total_offset,
        "last_log_likelihood_gain": float(last_gain),
        "minimum_log_likelihood_gain": minimum_gain,
        "max_weight_change": max_weight_change,
        "terminal_em_residual": terminal_em_residual,
        "weight_sum_error": weight_sum_error,
        "expected_count_total_error": expected_count_total_error,
        "posterior_sum_error": posterior_sum_error,
        "monotonic": monotonic,
        "n_observations": n_observations,
        "n_informative_observations": n_informative_observations,
        "n_uninformative_observations": n_uninformative_observations,
        "n_likelihood_rows": n_rows,
        "n_informative_rows": int(np.sum(informative_rows)),
        "n_uninformative_rows": int(np.sum(uninformative_rows)),
        "n_candidates": n_candidates,
        "n_zero_weight_candidates": int(np.sum(weights == 0)),
        "likelihood_support_density": support_entries / (n_rows * n_candidates),
        "block_size": block,
        "responsibilities_returned": bool(return_responsibilities),
    }
    return _KernelResult(
        weights=weights,
        expected_counts=expected_counts,
        responsibilities=responsibilities,
        log_likelihood_history=history_array,
        centered_log_likelihood_history=centered_history_array,
        converged=converged,
        n_iter=n_iter,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class EMResult:
    """Fixed-emission affinity fit and resolution-aware diagnostics."""

    weights: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    responsibilities: NDArray[np.float64] | None
    Q: NDArray[np.float64]
    aggregated: AggregatedEvidence
    observable_groups: tuple[tuple[int, ...], ...]
    log_likelihood_history: NDArray[np.float64]
    centered_log_likelihood_history: NDArray[np.float64]
    converged: bool
    n_iter: int
    diagnostics: dict[str, DiagnosticValue]

    @property
    def log_likelihood(self) -> float:
        return float(self.log_likelihood_history[-1])

    @property
    def analyzed_composition(self) -> NDArray[np.float64]:
        """Alias clarifying that ``weights`` describe the accepted traces."""

        return self.weights

    @property
    def observable_group_weights(self) -> NDArray[np.float64]:
        values = np.asarray(
            [self.weights[list(group)].sum() for group in self.observable_groups]
        )
        values.setflags(write=False)
        return values

    @property
    def observable_group_expected_counts(self) -> NDArray[np.float64]:
        values = np.asarray(
            [self.expected_counts[list(group)].sum() for group in self.observable_groups]
        )
        values.setflags(write=False)
        return values

    @property
    def observable_group_responsibilities(self) -> NDArray[np.float64] | None:
        if self.responsibilities is None:
            return None
        values = np.column_stack(
            [
                self.responsibilities[:, list(group)].sum(axis=1)
                for group in self.observable_groups
            ]
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class LikelihoodEMResult:
    """Mixture fit from an externally supplied observation-by-origin likelihood."""

    weights: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    responsibilities: NDArray[np.float64] | None
    log_likelihood_history: NDArray[np.float64]
    centered_log_likelihood_history: NDArray[np.float64]
    converged: bool
    n_iter: int
    diagnostics: dict[str, DiagnosticValue]

    @property
    def log_likelihood(self) -> float:
        return float(self.log_likelihood_history[-1])

    @property
    def analyzed_composition(self) -> NDArray[np.float64]:
        """Alias clarifying that ``weights`` describe the supplied records."""

        return self.weights


def fit_likelihood_em(
    log_likelihoods: Any,
    *,
    counts: Any | None = None,
    initial_weights: Any | None = None,
    max_iter: int = 1_000,
    tol: float = 1e-8,
    log_likelihood_offset: float = 0.0,
    block_size: int | None = 4096,
    return_responsibilities: bool = True,
) -> LikelihoodEMResult:
    """Estimate mixture weights from fixed per-class **log**-likelihood profiles.

    ``log_likelihoods`` is a class-by-origin matrix of ``log L_ik`` (use ``-inf``
    for structural zeros); ``counts`` gives each class's multiplicity.  The row
    scale cancels, so only relative values within a row matter.  Every
    positive-count row stays represented; rows equal across all origins add no
    mixture information and are handled analytically, with their counts restored
    at the fitted mixture.  No identification-confidence filter is applied.

    Examples
    --------
    >>> import numpy as np
    >>> # 3 trace-likelihood classes over 2 origins, with class multiplicities.
    >>> # NOTE: pass LOG-likelihoods, not likelihoods.
    >>> L = np.array([[0.9, 0.1],    # 60 classes favor origin 0 (9:1)
    ...               [0.1, 0.9],    # 30 favor origin 1
    ...               [0.5, 0.5]])   # 10 uninformative (equal under both)
    >>> fit = fit_likelihood_em(np.log(L), counts=[60, 30, 10])
    >>> fit.weights.round(3)
    array([0.708, 0.292])
    >>> fit.converged
    True
    """

    result = _fit_component_logs(
        log_likelihoods,
        counts=counts,
        initial_weights=initial_weights,
        max_iter=max_iter,
        tol=tol,
        log_likelihood_offset=log_likelihood_offset,
        block_size=block_size,
        return_responsibilities=return_responsibilities,
        copy_logs=True,
    )
    return LikelihoodEMResult(**result.__dict__)


def fit_em(
    observations: Any | AggregatedEvidence,
    profiles: Any | None = None,
    *,
    alpha: Any = 0.95,
    beta: Any = 0.05,
    Q: Any | None = None,
    cycle_to_probe: Any | None = None,
    counts: Any | None = None,
    initial_weights: Any | None = None,
    max_iter: int = 1_000,
    tol: float = 1e-8,
    block_size: int | None = 4096,
    return_responsibilities: bool = True,
) -> EMResult:
    """Fit fixed-emission affinity mixture weights from observations and ``Q``.

    ``observations`` is a molecule-by-cycle matrix of calls (``1`` positive,
    ``0`` negative, ``-1`` missing).  Supply emissions as a full ``Q`` or as
    binary ``profiles`` with ``alpha``/``beta``.  Without ``cycle_to_probe``,
    ``Q`` has one column per physical cycle (legacy).  With ``cycle_to_probe``,
    ``Q`` is per logical probe and the schedule maps cycles to columns; the
    fitter then builds exact repeated-probe count classes and factors out
    origin-independent probes, preserving their absolute log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([[0.90, 0.10, 0.80],    # 2 origins, 3 logical probes
    ...               [0.15, 0.85, 0.80]])
    >>> obs = np.array([[1, 0, 1],     # favors origin 0
    ...                 [0, 1, 1],     # favors origin 1
    ...                 [1, 0, 1]])    # favors origin 0
    >>> fit = fit_em(obs, Q=Q, cycle_to_probe=[0, 1, 2], return_responsibilities=False)
    >>> fit.weights.round(3)
    array([0.666, 0.334])
    >>> fit.converged
    True
    """

    q = build_emission_matrix(profiles, alpha=alpha, beta=beta, Q=Q)
    if isinstance(observations, ProbeCountClasses):
        if counts is not None or cycle_to_probe is not None:
            raise ValueError(
                "counts and cycle_to_probe must not be supplied with ProbeCountClasses"
            )
        if observations.n_logical_probes != q.shape[1]:
            raise ValueError("probe-count classes do not match the emission model")
        n_sufficient_classes = int(observations.source_n_classes)
        expected_indices = np.arange(observations.n_logical_probes)
        data = (
            compress_probe_count_classes(observations, q)
            if np.array_equal(observations.probe_indices, expected_indices)
            else observations
        )
        component_logs = log_likelihood_from_probe_counts(data, q)
        log_offset = data.log_likelihood_offset
        aggregation_rule = "repeated_probe_relative_likelihood"
        n_cycles = data.n_cycles
        n_logical_probes = data.n_logical_probes
        observed_columns = np.any(data.observed_counts > 0, axis=0)
        group_emissions = q[:, data.probe_indices[observed_columns]]
    elif cycle_to_probe is not None:
        raw = as_aggregated(observations, counts=counts)
        mapping = validate_cycle_to_probe(
            cycle_to_probe,
            n_cycles=raw.n_probes,
            n_logical_probes=q.shape[1],
        )
        sufficient = aggregate_probe_counts(
            raw,
            mapping,
            n_logical_probes=q.shape[1],
        )
        n_sufficient_classes = sufficient.n_unique
        data = compress_probe_count_classes(sufficient, q)
        component_logs = log_likelihood_from_probe_counts(data, q)
        log_offset = data.log_likelihood_offset
        aggregation_rule = "repeated_probe_relative_likelihood"
        n_cycles = raw.n_probes
        n_logical_probes = q.shape[1]
        observed_columns = np.any(data.observed_counts > 0, axis=0)
        group_emissions = q[:, data.probe_indices[observed_columns]]
    else:
        data = as_aggregated(observations, counts=counts)
        if data.n_probes != q.shape[1]:
            raise ValueError(
                f"observations have {data.n_probes} cycles; emission model has {q.shape[1]}"
            )
        component_logs = log_likelihood_from_aggregated(data, q)
        log_offset = 0.0
        n_sufficient_classes = data.n_unique
        aggregation_rule = "identical_trace_and_mask"
        n_cycles = data.n_probes
        n_logical_probes = q.shape[1]
        observed_columns = np.any(data.mask, axis=0)
        group_emissions = q[:, observed_columns]

    kernel = _fit_component_logs(
        component_logs,
        counts=data.counts,
        initial_weights=initial_weights,
        max_iter=max_iter,
        tol=tol,
        log_likelihood_offset=log_offset,
        block_size=block_size,
        return_responsibilities=return_responsibilities,
        copy_logs=False,
    )
    if group_emissions.shape[1] == 0:
        observable_groups = (tuple(range(q.shape[0])),)
    else:
        observable_groups = find_observable_groups(group_emissions).groups
    unresolved_groups = [group for group in observable_groups if len(group) > 1]

    q = np.asarray(q, dtype=float)
    q.setflags(write=False)
    diagnostics = dict(kernel.diagnostics)
    diagnostics.update(
        {
            "n_unique_traces": data.n_unique,
            "n_trace_likelihood_classes": data.n_unique,
            "n_sufficient_count_classes": n_sufficient_classes,
            "n_cycles": n_cycles,
            "n_probes": n_cycles,
            "n_logical_probes": n_logical_probes,
            "missing_fraction": data.missing_fraction,
            "aggregation_rule": aggregation_rule,
            "dense_entries_per_iteration": data.n_unique * q.shape[0],
            "n_observable_groups": len(observable_groups),
            "n_unresolved_groups": len(unresolved_groups),
            "n_candidates_in_unresolved_groups": int(
                sum(len(group) for group in unresolved_groups)
            ),
            "max_observable_group_size": max(map(len, observable_groups)),
        }
    )
    return EMResult(
        weights=kernel.weights,
        expected_counts=kernel.expected_counts,
        responsibilities=kernel.responsibilities,
        Q=q,
        aggregated=data,
        observable_groups=observable_groups,
        log_likelihood_history=kernel.log_likelihood_history,
        centered_log_likelihood_history=kernel.centered_log_likelihood_history,
        converged=kernel.converged,
        n_iter=kernel.n_iter,
        diagnostics=diagnostics,
    )


def posterior_responsibilities(
    observations: Any,
    weights: Any,
    Q: Any,
    *,
    cycle_to_probe: Any | None = None,
) -> NDArray[np.float64]:
    """Calculate posterior assignment probabilities for fixed mixture weights."""

    q = np.asarray(Q, dtype=float)
    if q.ndim != 2:
        raise ValueError("Q must be candidate-by-probe")
    normalized = _normalize_weights(weights, q.shape[0])
    if isinstance(observations, ProbeCountClasses):
        if cycle_to_probe is not None:
            raise ValueError("cycle_to_probe must not be supplied with ProbeCountClasses")
        logs = log_likelihood_from_probe_counts(observations, q)
        posterior, _ = _posterior_from_logs(logs, normalized)
        return posterior

    data = as_aggregated(observations)
    likelihood_q = q
    if cycle_to_probe is not None:
        mapping = validate_cycle_to_probe(
            cycle_to_probe,
            n_cycles=data.n_probes,
            n_logical_probes=q.shape[1],
        )
        likelihood_q = q[:, mapping]
    logs = log_likelihood_from_aggregated(data, likelihood_q)
    unique, _ = _posterior_from_logs(logs, normalized)
    if isinstance(observations, AggregatedTraces):
        return unique

    from .data import validate_observations

    raw = validate_observations(observations, n_probes=data.n_probes)
    lookup = {
        tuple(row.tolist()): posterior
        for row, posterior in zip(data.as_observations(), unique, strict=True)
    }
    return np.vstack([lookup[tuple(row.tolist())] for row in raw])
