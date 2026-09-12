"""Observation validation and sufficient-statistic aggregation.

The data layer behind the fixed-``Q`` EM fit. Observations are coded ``-1/0/1``,
where ``-1`` is a missing (NA) call that marginalizes to a neutral factor under
ignorable missingness (manuscript main Eq 4), never a third emission state.

Two exact, likelihood-preserving reductions keep the fit over a large candidate
panel tractable:

* :class:`AggregatedTraces` collapses identical trace-and-mask patterns to unique
  rows with multiplicities. A trace's likelihood depends only on its pattern, so
  molecules sharing a pattern share a row.
* :class:`ProbeCountClasses` summarizes cycles mapped to one logical probe by
  their positive and observed counts. Those counts are sufficient when the
  repeated cycles are conditionally independent and share one calibrated emission
  probability (the repeated-probe / class-compression reduction of Supplementary
  Methods S2.4). :func:`compress_probe_count_classes` then drops logical probes
  whose emission row is identical across every origin (e.g. the pan-tau probes):
  they contribute only an origin-independent factor, retained exactly as a
  log-likelihood offset, so assignment and mixture estimates are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


MISSING = -1


def validate_observations(
    observations: Any, *, n_probes: int | None = None
) -> NDArray[np.int8]:
    """Return a two-dimensional array containing only ``-1``, ``0``, and ``1``.

    ``-1`` denotes an unobserved probe cycle. It is a missingness marker, not a
    third emission state.
    """

    array = np.asarray(observations)
    if array.ndim != 2:
        raise ValueError("observations must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("observations must contain at least one trace and probe")
    if n_probes is not None and array.shape[1] != n_probes:
        raise ValueError(
            f"observations have {array.shape[1]} probes; expected {n_probes}"
        )
    if not np.all(np.isin(array, (MISSING, 0, 1))):
        raise ValueError("observations may contain only -1 (missing), 0, and 1")
    return np.asarray(array, dtype=np.int8)


def _validate_counts(counts: Any, n_rows: int) -> NDArray[np.int64]:
    array = np.asarray(counts)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise ValueError("counts must be a one-dimensional value per trace")
    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must be finite non-negative integers") from exc
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise ValueError("counts must be finite non-negative integers")
    integer = numeric.astype(np.int64)
    if integer.sum() <= 0:
        raise ValueError("at least one trace count must be positive")
    return integer


@dataclass(frozen=True)
class AggregatedTraces:
    """Unique binary traces, observation masks, and their multiplicities.

    Values in ``traces`` are binary. A ``False`` entry in ``mask`` means that
    the corresponding value was missing and must be marginalized. Missing
    positions are normalized to zero in ``traces`` so that a trace/mask pair
    has one canonical representation.
    """

    traces: NDArray[np.int8]
    mask: NDArray[np.bool_]
    counts: NDArray[np.int64]

    def __post_init__(self) -> None:
        traces = np.asarray(self.traces)
        mask = np.asarray(self.mask, dtype=bool)
        if traces.ndim != 2 or mask.shape != traces.shape:
            raise ValueError("traces and mask must be two-dimensional with equal shape")
        if traces.shape[0] == 0 or traces.shape[1] == 0:
            raise ValueError("aggregated traces must not be empty")
        if not np.all(np.isin(traces, (0, 1))):
            raise ValueError("aggregated trace values must be binary")
        counts = _validate_counts(self.counts, traces.shape[0])

        normalized = np.asarray(traces, dtype=np.int8).copy()
        normalized[~mask] = 0
        mask = mask.copy()
        counts = counts.copy()
        normalized.setflags(write=False)
        mask.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "traces", normalized)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "counts", counts)

    @property
    def n_unique(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_probes(self) -> int:
        """Compatibility alias for the number of physical cycle columns."""

        return int(self.traces.shape[1])

    @property
    def n_cycles(self) -> int:
        return int(self.traces.shape[1])

    @property
    def n_observations(self) -> int:
        return int(self.counts.sum())

    @property
    def missing_fraction(self) -> float:
        missing_per_trace = (~self.mask).sum(axis=1)
        missing = float(np.dot(self.counts, missing_per_trace))
        return missing / (self.n_observations * self.n_probes)

    def as_observations(self, *, expand: bool = False) -> NDArray[np.int8]:
        """Return ``-1/0/1`` observations, optionally expanding counts."""

        observations = np.where(self.mask, self.traces, MISSING).astype(np.int8)
        if expand:
            observations = np.repeat(observations, self.counts, axis=0)
        return observations


def aggregate_traces(
    observations: Any, counts: Any | None = None
) -> AggregatedTraces:
    """Collapse identical trace-and-mask patterns and sum their counts."""

    array = validate_observations(observations)
    row_counts = (
        np.ones(array.shape[0], dtype=np.int64)
        if counts is None
        else _validate_counts(counts, array.shape[0])
    )

    positive = row_counts > 0
    array = array[positive]
    row_counts = row_counts[positive]
    mask = array != MISSING
    traces = np.where(mask, array, 0).astype(np.int8)

    # Both parts are needed: the same zero-filled trace can represent a true
    # negative or a missing cycle, and those have different likelihoods.
    keys = np.concatenate((mask.astype(np.int8), traces), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    aggregated_counts = np.bincount(inverse, weights=row_counts).astype(np.int64)
    n_probes = array.shape[1]
    unique_mask = unique[:, :n_probes].astype(bool)
    unique_traces = unique[:, n_probes:].astype(np.int8)
    return AggregatedTraces(unique_traces, unique_mask, aggregated_counts)


def validate_cycle_to_probe(
    cycle_to_probe: Any,
    *,
    n_cycles: int,
    n_logical_probes: int | None = None,
) -> NDArray[np.int64]:
    """Validate the map from physical cycles to logical probes.

    The returned array has one zero-based logical-probe index per physical
    cycle.  Repeated indices identify exchangeable applications of the same
    calibrated logical probe.

    Examples
    --------
    >>> # four cycles alternating between logical probes 0 and 1
    >>> validate_cycle_to_probe([0, 1, 0, 1], n_cycles=4, n_logical_probes=2)
    array([0, 1, 0, 1])
    """

    raw = np.asarray(cycle_to_probe)
    if raw.ndim != 1 or raw.shape[0] != n_cycles:
        raise ValueError("cycle_to_probe must contain one index per physical cycle")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cycle_to_probe must contain non-negative integers") from exc
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise ValueError("cycle_to_probe must contain non-negative integers")
    mapping = numeric.astype(np.int64)
    if n_logical_probes is not None:
        if not isinstance(n_logical_probes, (int, np.integer)) or n_logical_probes < 1:
            raise ValueError("n_logical_probes must be a positive integer")
        if np.any(mapping >= int(n_logical_probes)):
            raise ValueError(
                "cycle_to_probe contains an index outside the logical-probe matrix"
            )
    mapping.setflags(write=False)
    return mapping


@dataclass(frozen=True)
class ProbeCountClasses:
    """Exact classes summarized by positive and observed calls per probe.

    These counts are sufficient when cycles mapped to one logical probe share
    the same calibrated emission probability and are conditionally
    independent. ``probe_indices`` records which columns of the full logical
    emission matrix remain assignment-informative. Columns whose emission is
    identical across every origin may be removed by
    :func:`compress_probe_count_classes`; their total contribution is retained
    in ``log_likelihood_offset``.
    """

    positive_counts: NDArray[np.int64]
    observed_counts: NDArray[np.int64]
    counts: NDArray[np.int64]
    probe_indices: NDArray[np.int64]
    n_logical_probes: int
    n_cycles: int
    missing_fraction: float
    log_likelihood_offset: float = 0.0
    source_n_classes: int | None = None

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive_counts)
        observed = np.asarray(self.observed_counts)
        if positive.ndim != 2 or observed.shape != positive.shape:
            raise ValueError(
                "positive_counts and observed_counts must be equal two-dimensional arrays"
            )
        if positive.shape[0] == 0:
            raise ValueError("probe-count classes must contain at least one row")
        try:
            positive_numeric = np.asarray(positive, dtype=float)
            observed_numeric = np.asarray(observed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe counts must be finite non-negative integers") from exc
        if (
            not np.all(np.isfinite(positive_numeric))
            or not np.all(np.isfinite(observed_numeric))
            or np.any(positive_numeric < 0)
            or np.any(observed_numeric < 0)
            or not np.all(positive_numeric == np.floor(positive_numeric))
            or not np.all(observed_numeric == np.floor(observed_numeric))
            or np.any(positive_numeric > observed_numeric)
        ):
            raise ValueError(
                "probe counts must be finite non-negative integers with positives <= observed"
            )
        if not isinstance(self.n_logical_probes, (int, np.integer)) or self.n_logical_probes < 1:
            raise ValueError("n_logical_probes must be a positive integer")
        if not isinstance(self.n_cycles, (int, np.integer)) or self.n_cycles < 1:
            raise ValueError("n_cycles must be a positive integer")
        indices = np.asarray(self.probe_indices)
        if indices.ndim != 1 or indices.shape[0] != positive.shape[1]:
            raise ValueError("probe_indices must identify every retained probe column")
        try:
            index_numeric = np.asarray(indices, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe_indices must contain non-negative integers") from exc
        if (
            not np.all(np.isfinite(index_numeric))
            or np.any(index_numeric < 0)
            or not np.all(index_numeric == np.floor(index_numeric))
        ):
            raise ValueError("probe_indices must contain non-negative integers")
        index_array = index_numeric.astype(np.int64)
        if np.any(index_array >= int(self.n_logical_probes)):
            raise ValueError("probe_indices exceed n_logical_probes")
        if np.unique(index_array).size != index_array.size:
            raise ValueError("probe_indices must not contain duplicates")
        if not np.isfinite(self.missing_fraction) or not 0 <= self.missing_fraction <= 1:
            raise ValueError("missing_fraction must be a probability")
        if not np.isfinite(self.log_likelihood_offset):
            raise ValueError("log_likelihood_offset must be finite")

        class_counts = _validate_counts(self.counts, positive.shape[0])
        source_n_classes = (
            positive.shape[0]
            if self.source_n_classes is None
            else int(self.source_n_classes)
        )
        if source_n_classes < positive.shape[0]:
            raise ValueError("source_n_classes cannot be smaller than the retained classes")

        positive_array = positive_numeric.astype(np.int64)
        observed_array = observed_numeric.astype(np.int64)
        for array in (positive_array, observed_array, class_counts, index_array):
            array.setflags(write=False)
        object.__setattr__(self, "positive_counts", positive_array)
        object.__setattr__(self, "observed_counts", observed_array)
        object.__setattr__(self, "counts", class_counts)
        object.__setattr__(self, "probe_indices", index_array)
        object.__setattr__(self, "n_logical_probes", int(self.n_logical_probes))
        object.__setattr__(self, "n_cycles", int(self.n_cycles))
        object.__setattr__(self, "missing_fraction", float(self.missing_fraction))
        object.__setattr__(
            self, "log_likelihood_offset", float(self.log_likelihood_offset)
        )
        object.__setattr__(self, "source_n_classes", source_n_classes)

    @property
    def n_unique(self) -> int:
        return int(self.positive_counts.shape[0])

    @property
    def n_observations(self) -> int:
        return int(self.counts.sum())

    @property
    def n_retained_probes(self) -> int:
        return int(self.positive_counts.shape[1])


def aggregate_probe_counts(
    observations: Any | AggregatedTraces,
    cycle_to_probe: Any,
    *,
    counts: Any | None = None,
    n_logical_probes: int | None = None,
) -> ProbeCountClasses:
    """Collapse traces into positive/observed counts for repeated probes.

    For each logical probe the sufficient statistic is ``(positive_counts,
    observed_counts)``, so traces that differ only in the *order* of a probe's
    repeated calls collapse to one class.  This aggregation is exact only when
    physical cycles mapped to the same logical probe share one calibrated
    emission probability.  Missing calls (``-1``) reduce the observed count and
    are never recoded as negatives.

    Examples
    --------
    >>> import numpy as np
    >>> # 4 cycles = probe 0 twice then probe 1 twice; the two rows differ only
    >>> # in the order of probe 0's calls, so they share one sufficient class.
    >>> obs = np.array([[1, 0, 1, 1],
    ...                 [0, 1, 1, 1]])
    >>> classes = aggregate_probe_counts(obs, [0, 0, 1, 1], n_logical_probes=2)
    >>> classes.n_unique
    1
    >>> classes.positive_counts, classes.observed_counts, classes.counts
    (array([[1, 2]]), array([[2, 2]]), array([2]))
    """

    if isinstance(observations, AggregatedTraces):
        if counts is not None:
            raise ValueError("counts must not be supplied with AggregatedTraces")
        array = observations.as_observations()
        row_counts = observations.counts
    else:
        array = validate_observations(observations)
        row_counts = (
            np.ones(array.shape[0], dtype=np.int64)
            if counts is None
            else _validate_counts(counts, array.shape[0])
        )

    positive_rows = row_counts > 0
    array = array[positive_rows]
    row_counts = row_counts[positive_rows]
    mapping = validate_cycle_to_probe(
        cycle_to_probe,
        n_cycles=array.shape[1],
        n_logical_probes=n_logical_probes,
    )
    logical_count = (
        int(mapping.max()) + 1 if n_logical_probes is None else int(n_logical_probes)
    )

    positive = np.zeros((array.shape[0], logical_count), dtype=np.int64)
    observed = np.zeros_like(positive)
    for probe in range(logical_count):
        selected = mapping == probe
        if not np.any(selected):
            continue
        calls = array[:, selected]
        positive[:, probe] = np.sum(calls == 1, axis=1)
        observed[:, probe] = np.sum(calls != MISSING, axis=1)

    keys = np.concatenate((observed, positive), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    aggregated_counts = np.bincount(inverse, weights=row_counts).astype(np.int64)
    missing_per_row = np.sum(array == MISSING, axis=1)
    missing_total = float(np.dot(row_counts, missing_per_row))
    n_observations = int(row_counts.sum())
    return ProbeCountClasses(
        positive_counts=unique[:, logical_count:],
        observed_counts=unique[:, :logical_count],
        counts=aggregated_counts,
        probe_indices=np.arange(logical_count, dtype=np.int64),
        n_logical_probes=logical_count,
        n_cycles=array.shape[1],
        missing_fraction=missing_total / (n_observations * array.shape[1]),
        source_n_classes=unique.shape[0],
    )


def compress_probe_count_classes(
    data: ProbeCountClasses, Q: Any
) -> ProbeCountClasses:
    """Remove origin-independent probe factors from assignment classes
    (the class-compression reduction of Supplementary Methods S2.4).

    A logical probe whose calibrated positive-call probability is exactly the
    same for every candidate contributes only a row-wide likelihood factor, so
    its outcomes are dropped from posterior assignment and EM updates.  Their
    summed contribution is retained exactly as an absolute log-likelihood
    offset, so the total log-likelihood is unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from proteoem import aggregate_probe_counts
    >>> obs = np.array([[1, 0, 1, 1], [0, 1, 1, 1]])
    >>> classes = aggregate_probe_counts(obs, [0, 0, 1, 1], n_logical_probes=2)
    >>> # logical probe 1 has an identical column for both origins -> uninformative
    >>> Q = np.array([[0.9, 0.5],
    ...               [0.2, 0.5]])
    >>> relative = compress_probe_count_classes(classes, Q)
    >>> relative.probe_indices                            # only probe 0 is kept
    array([0])
    >>> round(float(relative.log_likelihood_offset), 4)   # probe 1's constant factor
    -2.7726
    """

    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[1] != data.n_logical_probes:
        raise ValueError("Q must be candidate-by-logical-probe")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q probabilities must be finite and in [0, 1]")
    expected_indices = np.arange(data.n_logical_probes, dtype=np.int64)
    if not np.array_equal(data.probe_indices, expected_indices):
        raise ValueError("probe-count classes have already been compressed")

    invariant = np.all(q == q[[0], :], axis=0)
    retained_indices = np.flatnonzero(~invariant).astype(np.int64)
    offset_total = float(data.log_likelihood_offset)

    for probe in np.flatnonzero(invariant):
        probability = float(q[0, probe])
        positives = data.positive_counts[:, probe]
        negatives = data.observed_counts[:, probe] - positives
        impossible = ((probability == 0.0) & (positives > 0)) | (
            (probability == 1.0) & (negatives > 0)
        )
        if np.any(impossible):
            rows = np.flatnonzero(impossible).tolist()
            raise ValueError(
                "some probe-count patterns have zero probability under every candidate "
                f"(rows {rows})"
            )
        row_offsets = np.zeros(data.n_unique, dtype=float)
        if probability > 0.0:
            row_offsets += positives * np.log(probability)
        if probability < 1.0:
            row_offsets += negatives * np.log1p(-probability)
        offset_total += float(np.dot(data.counts, row_offsets))

    positive = data.positive_counts[:, retained_indices]
    observed = data.observed_counts[:, retained_indices]
    if retained_indices.size == 0:
        inverse = np.zeros(data.n_unique, dtype=np.int64)
        unique_observed = np.zeros((1, 0), dtype=np.int64)
        unique_positive = np.zeros((1, 0), dtype=np.int64)
    else:
        keys = np.concatenate((observed, positive), axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        width = retained_indices.size
        unique_observed = unique[:, :width]
        unique_positive = unique[:, width:]
    compressed_counts = np.bincount(inverse, weights=data.counts).astype(np.int64)
    return ProbeCountClasses(
        positive_counts=unique_positive,
        observed_counts=unique_observed,
        counts=compressed_counts,
        probe_indices=retained_indices,
        n_logical_probes=data.n_logical_probes,
        n_cycles=data.n_cycles,
        missing_fraction=data.missing_fraction,
        log_likelihood_offset=offset_total,
        source_n_classes=data.source_n_classes,
    )


def as_aggregated(
    observations: Any | AggregatedTraces, counts: Any | None = None
) -> AggregatedTraces:
    """Normalize raw or already-aggregated input to :class:`AggregatedTraces`."""

    if isinstance(observations, AggregatedTraces):
        if counts is not None:
            raise ValueError("counts must not be supplied with AggregatedTraces")
        return observations
    return aggregate_traces(observations, counts=counts)
