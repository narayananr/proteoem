"""Reusable utilities for the fixed-Q model-violation benchmark (manuscript
Section 3.3 and Fig. 7).

The helpers here keep simulation policy out of the core estimator. They provide:

* :func:`build_trace_likelihood_classes` -- exact compression to the
  trace-likelihood classes of main Eq 11. Rows differing only by an
  origin-independent factor are merged; the dropped constant is kept as an offset
  so absolute predictive likelihoods still audit.
* :func:`probability_assignment_metrics` -- the probabilistic assignment scores
  reported in the benchmark (accuracy, log score / NLL, Brier, and the
  calibration gap / ECE and overconfidence), evaluated both per candidate and per
  observable proteoform group (main Eq 10).
* :func:`bootstrap_fixed_q_classes` -- a nonparametric bootstrap over those
  classes; intervals are conditional on the supplied likelihood matrix, and
  therefore on fixed ``Q``. The ``NA_as_zero_hard_ml`` estimator is supplied only
  as the declared hard-call negative control (a reduced baseline, Supplementary
  Methods S2.5).
* :func:`emission_distribution_diagnostics` -- a near-identifiability diagnostic
  (Supplementary Methods S2.7): how well separated the per-origin outcome
  distributions are, summarized analytically without enumerating ``2**C`` traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .data import ProbeCountClasses, validate_cycle_to_probe, validate_observations
from .em import fit_likelihood_em, logsumexp
from .emissions import log_likelihood_from_probe_counts


@dataclass(frozen=True)
class TraceLikelihoodClasses:
    """Exact relative-likelihood rows and their observed multiplicities.

    ``molecule_to_class`` maps every input molecule to a row in
    ``log_likelihoods``.  The rows have been centered by an origin-independent
    constant and then deduplicated.  ``molecule_log_offsets`` retains those
    constants so absolute predictive likelihoods can still be audited.
    """

    log_likelihoods: NDArray[np.float64]
    counts: NDArray[np.int64]
    molecule_to_class: NDArray[np.int64]
    molecule_log_offsets: NDArray[np.float64]
    log_likelihood_offset: float
    missing_fraction: float

    def __post_init__(self) -> None:
        logs = np.asarray(self.log_likelihoods, dtype=float)
        counts = np.asarray(self.counts)
        membership = np.asarray(self.molecule_to_class)
        offsets = np.asarray(self.molecule_log_offsets, dtype=float)
        if logs.ndim != 2 or min(logs.shape) < 1:
            raise ValueError("log_likelihoods must be a non-empty matrix")
        if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise ValueError("log_likelihoods may contain finite values or -inf")
        if counts.ndim != 1 or counts.shape[0] != logs.shape[0]:
            raise ValueError("counts must contain one value per likelihood class")
        if membership.ndim != 1 or offsets.shape != membership.shape:
            raise ValueError("molecule mappings and offsets must be equal vectors")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("class counts must be positive integers")
        if not np.issubdtype(membership.dtype, np.integer):
            raise ValueError("molecule_to_class must contain integers")
        if np.any((membership < 0) | (membership >= logs.shape[0])):
            raise ValueError("molecule_to_class contains an invalid class index")
        if int(counts.sum()) != membership.size:
            raise ValueError("class counts must sum to the number of molecules")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("molecule log offsets must be finite")
        if not np.isfinite(self.log_likelihood_offset):
            raise ValueError("log_likelihood_offset must be finite")
        if not np.isfinite(self.missing_fraction) or not 0 <= self.missing_fraction <= 1:
            raise ValueError("missing_fraction must be a probability")

        logs = logs.copy()
        counts = counts.astype(np.int64, copy=True)
        membership = membership.astype(np.int64, copy=True)
        offsets = offsets.copy()
        for array in (logs, counts, membership, offsets):
            array.setflags(write=False)
        object.__setattr__(self, "log_likelihoods", logs)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "molecule_to_class", membership)
        object.__setattr__(self, "molecule_log_offsets", offsets)
        object.__setattr__(self, "log_likelihood_offset", float(self.log_likelihood_offset))
        object.__setattr__(self, "missing_fraction", float(self.missing_fraction))

    @property
    def n_classes(self) -> int:
        return int(self.log_likelihoods.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.log_likelihoods.shape[1])

    @property
    def n_molecules(self) -> int:
        return int(self.molecule_to_class.size)


def build_trace_likelihood_classes(
    observations: Any,
    Q: Any,
    *,
    cycle_to_probe: Any | None = None,
) -> TraceLikelihoodClasses:
    """Compress observations into the trace-likelihood classes of main Eq 11.

    Observations with the same relative-likelihood profile under ``Q`` form one
    class.  Repeated physical cycles can be mapped to logical probes.  Positive and
    observed counts are then sufficient for the independent Bernoulli decoder.
    Missing calls reduce the observed count; they are never recoded as zero.
    """

    array = validate_observations(observations)
    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or min(q.shape) < 1:
        raise ValueError("Q must be a non-empty candidate-by-probe matrix")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q probabilities must be finite and in [0, 1]")

    if cycle_to_probe is None:
        if q.shape[1] != array.shape[1]:
            raise ValueError("Q must have one column per observation cycle")
        mapping = np.arange(array.shape[1], dtype=np.int64)
    else:
        mapping = validate_cycle_to_probe(
            cycle_to_probe,
            n_cycles=array.shape[1],
            n_logical_probes=q.shape[1],
        )

    n_molecules = array.shape[0]
    n_logical = q.shape[1]
    positive = np.zeros((n_molecules, n_logical), dtype=np.int64)
    observed = np.zeros_like(positive)
    for probe in range(n_logical):
        selected = mapping == probe
        if not np.any(selected):
            continue
        calls = array[:, selected]
        positive[:, probe] = np.sum(calls == 1, axis=1)
        observed[:, probe] = np.sum(calls != -1, axis=1)

    keys = np.concatenate((observed, positive), axis=1)
    unique, molecule_to_count_class, count_class_counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    evidence = ProbeCountClasses(
        positive_counts=unique[:, n_logical:],
        observed_counts=unique[:, :n_logical],
        counts=count_class_counts,
        probe_indices=np.arange(n_logical, dtype=np.int64),
        n_logical_probes=n_logical,
        n_cycles=array.shape[1],
        missing_fraction=float(np.mean(array == -1)),
    )
    absolute_logs = log_likelihood_from_probe_counts(evidence, q)
    row_offsets = np.max(absolute_logs, axis=1)
    if np.any(~np.isfinite(row_offsets)):
        bad = np.flatnonzero(~np.isfinite(row_offsets)).tolist()
        raise ValueError(
            "some observed patterns have zero probability under every candidate "
            f"(count classes {bad})"
        )
    relative_logs = absolute_logs - row_offsets[:, None]

    # Different sufficient-count rows can differ only by an
    # origin-independent factor (for example, a pan-probe outcome).  Merging
    # them is exact for assignment and mixture estimation.
    likelihood_logs, count_to_likelihood, _ = np.unique(
        relative_logs,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    likelihood_counts = np.bincount(
        count_to_likelihood,
        weights=count_class_counts,
        minlength=likelihood_logs.shape[0],
    ).astype(np.int64)
    molecule_to_class = count_to_likelihood[molecule_to_count_class]
    molecule_offsets = row_offsets[molecule_to_count_class]
    total_offset = float(np.dot(count_class_counts, row_offsets))
    return TraceLikelihoodClasses(
        log_likelihoods=likelihood_logs,
        counts=likelihood_counts,
        molecule_to_class=molecule_to_class,
        molecule_log_offsets=molecule_offsets,
        log_likelihood_offset=total_offset,
        missing_fraction=evidence.missing_fraction,
    )


def _validate_groups(
    groups: Sequence[Sequence[int]] | None, n_candidates: int
) -> tuple[tuple[int, ...], ...]:
    if groups is None:
        return tuple((candidate,) for candidate in range(n_candidates))
    normalized = tuple(tuple(int(item) for item in group) for group in groups)
    flat = [item for group in normalized for item in group]
    if not normalized or any(not group for group in normalized):
        raise ValueError("observable groups must be non-empty")
    if sorted(flat) != list(range(n_candidates)):
        raise ValueError("observable groups must partition candidate indices")
    return normalized


@dataclass(frozen=True)
class FixedQBootstrapResult:
    """Trace-likelihood-class bootstrap replicates conditional on fixed Q."""

    candidate_weights: NDArray[np.float64]
    group_weights: NDArray[np.float64]
    converged: NDArray[np.bool_]
    failure_messages: tuple[str, ...]
    interval_level: float
    scope: str = "conditional_on_fixed_Q"

    @property
    def n_successful(self) -> int:
        return int(np.sum(self.converged))

    @property
    def n_failed(self) -> int:
        return int(self.converged.size - self.n_successful)

    def candidate_interval(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _percentile_interval(self.candidate_weights, self.interval_level)

    def group_interval(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _percentile_interval(self.group_weights, self.interval_level)


def _percentile_interval(
    replicates: NDArray[np.float64], interval_level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    successful = np.all(np.isfinite(replicates), axis=1)
    if not np.any(successful):
        width = replicates.shape[1]
        return np.full(width, np.nan), np.full(width, np.nan)
    tail = (1.0 - interval_level) * 50.0
    lower, upper = np.percentile(replicates[successful], [tail, 100.0 - tail], axis=0)
    return np.asarray(lower), np.asarray(upper)


def bootstrap_fixed_q_classes(
    log_likelihoods: Any,
    counts: Any,
    *,
    n_bootstrap: int,
    seed: int,
    groups: Sequence[Sequence[int]] | None = None,
    estimator: str = "em_fixed_emissions",
    initial_weights: Any | None = None,
    interval_level: float = 0.95,
    max_iter: int = 500,
    tol: float = 1e-8,
    block_size: int | None = 4096,
) -> FixedQBootstrapResult:
    """Resample likelihood-class multiplicities and refit at fixed Q.

    ``estimator='NA_as_zero_hard_ml'`` is supported solely for the declared
    negative control; it uses deterministic first-index tie breaking.
    Non-converged EM replicates are retained as explicit failed rows (NaNs),
    not silently included in percentile intervals.
    """

    logs = np.asarray(log_likelihoods, dtype=float)
    multiplicities = np.asarray(counts)
    if logs.ndim != 2 or min(logs.shape) < 1:
        raise ValueError("log_likelihoods must be a non-empty matrix")
    if multiplicities.ndim != 1 or multiplicities.shape[0] != logs.shape[0]:
        raise ValueError("counts must contain one value per likelihood row")
    numeric_counts = np.asarray(multiplicities, dtype=float)
    if (
        not np.all(np.isfinite(numeric_counts))
        or np.any(numeric_counts < 0)
        or not np.all(numeric_counts == np.floor(numeric_counts))
        or numeric_counts.sum() <= 0
    ):
        raise ValueError("counts must be non-negative integers with positive sum")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(interval_level) or not 0 < interval_level < 1:
        raise ValueError("interval_level must lie strictly between zero and one")
    if estimator not in {"em_fixed_emissions", "NA_as_zero_hard_ml"}:
        raise ValueError("unsupported bootstrap estimator")

    bootstrap_initial: NDArray[np.float64] | None = None
    if initial_weights is not None:
        bootstrap_initial = np.asarray(initial_weights, dtype=float)
        if (
            bootstrap_initial.ndim != 1
            or bootstrap_initial.shape[0] != logs.shape[1]
            or not np.all(np.isfinite(bootstrap_initial))
            or np.any(bootstrap_initial < 0)
            or bootstrap_initial.sum() <= 0
        ):
            raise ValueError(
                "initial_weights must be finite, non-negative, and match candidates"
            )
        bootstrap_initial = bootstrap_initial / bootstrap_initial.sum()

    integer_counts = numeric_counts.astype(np.int64)
    n_observations = int(integer_counts.sum())
    probabilities = integer_counts / n_observations
    normalized_groups = _validate_groups(groups, logs.shape[1])
    rng = np.random.default_rng(seed)
    candidates = np.full((int(n_bootstrap), logs.shape[1]), np.nan)
    grouped = np.full((int(n_bootstrap), len(normalized_groups)), np.nan)
    converged = np.zeros(int(n_bootstrap), dtype=bool)
    messages = [""] * int(n_bootstrap)

    hard_labels = np.argmax(logs, axis=1)
    for replicate in range(int(n_bootstrap)):
        sampled = rng.multinomial(n_observations, probabilities)
        try:
            if estimator == "NA_as_zero_hard_ml":
                weights = np.bincount(
                    hard_labels,
                    weights=sampled,
                    minlength=logs.shape[1],
                ).astype(float)
                weights /= weights.sum()
                ok = True
            else:
                fit = fit_likelihood_em(
                    logs,
                    counts=sampled,
                    initial_weights=bootstrap_initial,
                    max_iter=max_iter,
                    tol=tol,
                    block_size=block_size,
                    return_responsibilities=False,
                )
                weights = np.asarray(fit.weights)
                ok = bool(fit.converged)
                if not ok:
                    messages[replicate] = "EM did not converge"
            if ok:
                candidates[replicate] = weights
                grouped[replicate] = [
                    float(np.sum(weights[list(group)])) for group in normalized_groups
                ]
                converged[replicate] = True
        except (FloatingPointError, ValueError) as exc:
            messages[replicate] = f"{type(exc).__name__}: {exc}"

    for array in (candidates, grouped, converged):
        array.setflags(write=False)
    return FixedQBootstrapResult(
        candidate_weights=candidates,
        group_weights=grouped,
        converged=converged,
        failure_messages=tuple(messages),
        interval_level=float(interval_level),
    )


def _posterior(logs: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        joint = logs + np.log(weights)[None, :]
    normalizer = np.asarray(logsumexp(joint, axis=1))
    # A row whose entire finite-likelihood support has weight exactly 0 gives a
    # -inf normalizer; ``exp(-inf - (-inf))`` is NaN, which would silently
    # poison every downstream score.  Reject that degenerate input explicitly.
    if not np.all(np.isfinite(normalizer)):
        raise ValueError(
            "supplied weights give zero total evidence for at least one observed "
            "trace-likelihood class; its posterior is undefined"
        )
    posterior = np.exp(joint - normalizer[:, None])
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior


def _empty_reliability(n_bins: int) -> dict[str, NDArray]:
    return {
        "count": np.zeros(n_bins, dtype=np.int64),
        "confidence_sum": np.zeros(n_bins, dtype=float),
        "correct_sum": np.zeros(n_bins, dtype=float),
    }


def _add_reliability(
    accumulator: dict[str, NDArray],
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    counts: NDArray[np.int64],
) -> None:
    n_bins = accumulator["count"].size
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    accumulator["count"] += np.bincount(bins, weights=counts, minlength=n_bins).astype(
        np.int64
    )
    accumulator["confidence_sum"] += np.bincount(
        bins,
        weights=counts * confidence,
        minlength=n_bins,
    )
    accumulator["correct_sum"] += np.bincount(
        bins,
        weights=counts * correct,
        minlength=n_bins,
    )


def _finish_reliability(
    accumulator: dict[str, NDArray], *, level: str
) -> tuple[list[dict[str, float | int | str]], float]:
    total = int(accumulator["count"].sum())
    rows: list[dict[str, float | int | str]] = []
    ece = 0.0
    n_bins = accumulator["count"].size
    for index in range(n_bins):
        count = int(accumulator["count"][index])
        mean_confidence = (
            float(accumulator["confidence_sum"][index] / count) if count else np.nan
        )
        accuracy = float(accumulator["correct_sum"][index] / count) if count else np.nan
        gap = abs(mean_confidence - accuracy) if count else np.nan
        if total and count:
            ece += (count / total) * gap
        rows.append(
            {
                "level": level,
                "bin_index": index,
                "bin_lower": index / n_bins,
                "bin_upper": (index + 1) / n_bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return rows, float(ece) if total else np.nan


def _finish_equal_frequency_reliability(
    confidence_parts: Sequence[NDArray[np.float64]],
    correct_parts: Sequence[NDArray[np.bool_]],
    count_parts: Sequence[NDArray[np.int64]],
    *,
    level: str,
    n_bins: int,
) -> tuple[list[dict[str, float | int | str]], float]:
    if not confidence_parts:
        return (
            [
                {
                    "level": level,
                    "bin_index": index,
                    "bin_lower": np.nan,
                    "bin_upper": np.nan,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                    "absolute_gap": np.nan,
                }
                for index in range(n_bins)
            ],
            np.nan,
        )
    confidence = np.repeat(
        np.concatenate(confidence_parts), np.concatenate(count_parts)
    )
    correct = np.repeat(np.concatenate(correct_parts), np.concatenate(count_parts))
    order = np.argsort(confidence, kind="mergesort")
    partitions = np.array_split(order, n_bins)
    rows: list[dict[str, float | int | str]] = []
    ece = 0.0
    for index, selected in enumerate(partitions):
        count = int(selected.size)
        if count:
            selected_confidence = confidence[selected]
            mean_confidence = float(np.mean(selected_confidence))
            accuracy = float(np.mean(correct[selected]))
            gap = abs(mean_confidence - accuracy)
            lower = float(np.min(selected_confidence))
            upper = float(np.max(selected_confidence))
            ece += (count / confidence.size) * gap
        else:
            mean_confidence = accuracy = gap = lower = upper = np.nan
        rows.append(
            {
                "level": level,
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return rows, float(ece)


def probability_assignment_metrics(
    classes: TraceLikelihoodClasses,
    weights: Any,
    true_identities: Any,
    *,
    true_to_decoder: Any | None = None,
    observable_groups: Sequence[Sequence[int]] | None = None,
    n_bins: int = 10,
    binning: str = "equal_width",
    block_size: int = 2048,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Score candidate and observable-group posteriors without N-by-K expansion.

    Reports the benchmark's probabilistic assignment quality (Section 3.3): the
    posterior over origins (E-step responsibilities, main Eq 7) is scored both per
    candidate and per observable proteoform group (main Eq 10), giving accuracy,
    log score / NLL, Brier, and the calibration gap (ECE) and overconfidence.

    Truth states mapped to ``-1`` are excluded from assignment scores, as is
    necessary for omitted-origin closed-set fits.  Reliability rows use
    equal-width bins of the maximum posterior probability.
    """

    fitted_weights = np.asarray(weights, dtype=float)
    if fitted_weights.ndim != 1 or fitted_weights.shape[0] != classes.n_candidates:
        raise ValueError("weights must contain one value per decoder candidate")
    if (
        not np.all(np.isfinite(fitted_weights))
        or np.any(fitted_weights < 0)
        or fitted_weights.sum() <= 0
    ):
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    fitted_weights = fitted_weights / fitted_weights.sum()
    identities = np.asarray(true_identities)
    if identities.ndim != 1 or identities.shape[0] != classes.n_molecules:
        raise ValueError("true_identities must contain one value per molecule")
    if not np.issubdtype(identities.dtype, np.integer) or np.any(identities < 0):
        raise ValueError("true_identities must be non-negative integers")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    if binning not in {"equal_width", "equal_frequency"}:
        raise ValueError("binning must be 'equal_width' or 'equal_frequency'")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be positive")

    n_truth = int(identities.max()) + 1
    if true_to_decoder is None:
        mapping = np.arange(n_truth, dtype=np.int64)
    else:
        mapping = np.asarray(true_to_decoder)
        if mapping.ndim != 1 or mapping.shape[0] < n_truth:
            raise ValueError("true_to_decoder must map every observed truth identity")
        if not np.issubdtype(mapping.dtype, np.integer):
            raise ValueError("true_to_decoder must contain integer indices")
        mapping = mapping.astype(np.int64, copy=False)
    if np.any(mapping >= classes.n_candidates):
        raise ValueError("true_to_decoder contains an out-of-range decoder index")

    groups = _validate_groups(observable_groups, classes.n_candidates)
    candidate_to_group = np.empty(classes.n_candidates, dtype=np.int64)
    for group_index, group in enumerate(groups):
        candidate_to_group[list(group)] = group_index

    pairs, pair_counts = np.unique(
        np.column_stack((classes.molecule_to_class, identities.astype(np.int64))),
        axis=0,
        return_counts=True,
    )
    pair_class = pairs[:, 0]
    pair_truth = pairs[:, 1]
    pair_decoder = mapping[pair_truth]

    accumulators = {
        "candidate": {
            "n": 0,
            "log": 0.0,
            "brier": 0.0,
            "correct": 0.0,
            "confidence": 0.0,
        },
        "group": {
            "n": 0,
            "log": 0.0,
            "brier": 0.0,
            "correct": 0.0,
            "confidence": 0.0,
        },
    }
    reliability = {
        "candidate": _empty_reliability(int(n_bins)),
        "group": _empty_reliability(int(n_bins)),
    }
    equal_frequency_parts: dict[str, dict[str, list[NDArray]]] = {
        "candidate": {"confidence": [], "correct": [], "counts": []},
        "group": {"confidence": [], "correct": [], "counts": []},
    }
    tiny = np.finfo(float).tiny

    for start in range(0, classes.n_classes, int(block_size)):
        stop = min(start + int(block_size), classes.n_classes)
        posterior = _posterior(classes.log_likelihoods[start:stop], fitted_weights)
        group_posterior = np.column_stack(
            [posterior[:, list(group)].sum(axis=1) for group in groups]
        )
        left = int(np.searchsorted(pair_class, start, side="left"))
        right = int(np.searchsorted(pair_class, stop, side="left"))
        if left == right:
            continue
        local_rows = pair_class[left:right] - start
        decoder = pair_decoder[left:right]
        counts = pair_counts[left:right].astype(np.int64)
        valid = decoder >= 0
        if not np.any(valid):
            continue

        local_rows = local_rows[valid]
        decoder = decoder[valid]
        counts = counts[valid]
        selected = posterior[local_rows]
        probability_true = selected[np.arange(selected.shape[0]), decoder]
        top = np.argmax(selected, axis=1)
        confidence = np.max(selected, axis=1)
        correct = top == decoder
        candidate_acc = accumulators["candidate"]
        candidate_acc["n"] += int(counts.sum())
        candidate_acc["log"] += float(np.dot(counts, np.log(np.maximum(probability_true, tiny))))
        brier = np.sum(selected * selected, axis=1) - 2.0 * probability_true + 1.0
        candidate_acc["brier"] += float(np.dot(counts, brier))
        candidate_acc["correct"] += float(np.dot(counts, correct))
        candidate_acc["confidence"] += float(np.dot(counts, confidence))
        _add_reliability(reliability["candidate"], confidence, correct, counts)
        equal_frequency_parts["candidate"]["confidence"].append(confidence)
        equal_frequency_parts["candidate"]["correct"].append(correct)
        equal_frequency_parts["candidate"]["counts"].append(counts)

        true_group = candidate_to_group[decoder]
        selected_groups = group_posterior[local_rows]
        probability_group = selected_groups[
            np.arange(selected_groups.shape[0]), true_group
        ]
        top_group = np.argmax(selected_groups, axis=1)
        group_confidence = np.max(selected_groups, axis=1)
        correct_group = top_group == true_group
        group_acc = accumulators["group"]
        group_acc["n"] += int(counts.sum())
        group_acc["log"] += float(
            np.dot(counts, np.log(np.maximum(probability_group, tiny)))
        )
        group_brier = (
            np.sum(selected_groups * selected_groups, axis=1)
            - 2.0 * probability_group
            + 1.0
        )
        group_acc["brier"] += float(np.dot(counts, group_brier))
        group_acc["correct"] += float(np.dot(counts, correct_group))
        group_acc["confidence"] += float(np.dot(counts, group_confidence))
        _add_reliability(
            reliability["group"], group_confidence, correct_group, counts
        )
        equal_frequency_parts["group"]["confidence"].append(group_confidence)
        equal_frequency_parts["group"]["correct"].append(correct_group)
        equal_frequency_parts["group"]["counts"].append(counts)

    metrics: dict[str, float | int] = {}
    rows: list[dict[str, float | int | str]] = []
    for level in ("candidate", "group"):
        accumulator = accumulators[level]
        n_evaluated = int(accumulator["n"])
        if binning == "equal_frequency":
            parts = equal_frequency_parts[level]
            reliability_rows, ece = _finish_equal_frequency_reliability(
                parts["confidence"],
                parts["correct"],
                parts["counts"],
                level=level,
                n_bins=int(n_bins),
            )
        else:
            reliability_rows, ece = _finish_reliability(
                reliability[level], level=level
            )
        rows.extend(reliability_rows)
        metrics[f"{level}_assignment_n"] = n_evaluated
        if n_evaluated:
            mean_log = float(accumulator["log"] / n_evaluated)
            accuracy = float(accumulator["correct"] / n_evaluated)
            mean_confidence = float(accumulator["confidence"] / n_evaluated)
            metrics[f"{level}_assignment_log_score"] = mean_log
            metrics[f"{level}_assignment_nll"] = -mean_log
            metrics[f"{level}_assignment_brier"] = float(
                accumulator["brier"] / n_evaluated
            )
            metrics[f"{level}_assignment_accuracy"] = accuracy
            metrics[f"{level}_assignment_mean_confidence"] = mean_confidence
            metrics[f"{level}_assignment_ece"] = ece
            metrics[f"{level}_assignment_overconfidence"] = (
                mean_confidence - accuracy
            )
        else:
            for suffix in (
                "log_score",
                "nll",
                "brier",
                "accuracy",
                "mean_confidence",
                "ece",
                "overconfidence",
            ):
                metrics[f"{level}_assignment_{suffix}"] = np.nan
    metrics["assignment_excluded_n"] = classes.n_molecules - int(
        accumulators["candidate"]["n"]
    )
    metrics["assignment_coverage"] = float(
        accumulators["candidate"]["n"] / classes.n_molecules
    )
    return metrics, rows


def emission_distribution_diagnostics(
    Q: Any, *, rank_tolerance: float = 1e-10
) -> dict[str, float | int]:
    """Diagnose separation of independent-Bernoulli component distributions
    (the near-identifiability diagnostic of Supplementary Methods S2.7).

    The Gram matrix is computed analytically over the complete binary outcome
    space: ``G[k,l] = sum_y p(y|k)p(y|l)``.  Its eigenvalues are squared
    singular values of the candidate-by-outcome probability matrix, avoiding
    explicit enumeration of ``2**n_cycles`` traces.
    """

    q = np.asarray(Q, dtype=float)
    # ``rank_tolerance`` is applied in eigenvalue (squared-singular-value) space,
    # see the rank test below.
    if q.ndim != 2 or min(q.shape) < 1:
        raise ValueError("Q must be a non-empty candidate-by-cycle matrix")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q probabilities must be finite and in [0, 1]")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be positive")

    n_candidates = q.shape[0]
    gram = np.ones((n_candidates, n_candidates), dtype=float)
    bhattacharyya = np.ones_like(gram)
    for cycle in range(q.shape[1]):
        values = q[:, cycle]
        gram *= np.outer(values, values) + np.outer(1.0 - values, 1.0 - values)
        bhattacharyya *= np.sqrt(np.outer(values, values)) + np.sqrt(
            np.outer(1.0 - values, 1.0 - values)
        )
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_max = float(eigenvalues[-1])
    # Decide rank from the eigenvalues (the squared singular values), NOT from
    # their square roots.  Forming ``G = P @ P.T`` squares the dynamic range, so
    # a genuine zero singular value of ``P`` leaves an eigenvalue at only
    # ``~eps * lambda_max``; thresholding here keeps that rounding noise far
    # below the cutoff.  Taking the sqrt first would lift the same noise to
    # ``~sqrt(eps) * sigma_max`` and silently mask an exact rank drop (reporting
    # full rank, and a finite condition number, for a degenerate panel).
    positive = eigenvalues > rank_tolerance * lambda_max
    rank = int(np.sum(positive))
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    maximum = float(singular_values[-1])
    minimum_retained = float(singular_values[positive][0]) if rank else 0.0
    condition = maximum / minimum_retained if rank else np.inf
    if rank < n_candidates:
        condition = np.inf

    np.fill_diagonal(bhattacharyya, -np.inf)
    closest_flat = int(np.argmax(bhattacharyya)) if n_candidates > 1 else 0
    first, second = np.unravel_index(closest_flat, bhattacharyya.shape)
    closest_bc = float(bhattacharyya[first, second]) if n_candidates > 1 else 1.0
    minimum_hellinger = float(np.sqrt(max(0.0, 1.0 - min(1.0, closest_bc))))
    return {
        "component_distribution_rank": rank,
        "component_distribution_candidates": n_candidates,
        "component_singular_value_max": maximum,
        "component_singular_value_min_retained": minimum_retained,
        "component_condition_number": float(condition),
        "minimum_pair_hellinger": minimum_hellinger,
        "closest_pair_first": int(first),
        "closest_pair_second": int(second),
    }
