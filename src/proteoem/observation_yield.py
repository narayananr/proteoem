"""Source-to-observation corrections for intact-molecule affinity traces.

The observation-yield layer of manuscript Section 2.3. The fixed-emission EM
kernel estimates the composition ``pi`` among *accepted* traces; this module maps
between that and the *source* composition ``theta``, in a separate, auditable
layer. The effective yield ``e_k = r_k * v_k`` (physical recovery times gate
visibility) is the accepted-trace probability per source molecule, and
``theta_k`` is proportional to ``pi_k / e_k``. Retention conditioning on the gate
and inverse-yield correction do different jobs and must each be applied once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import validate_cycle_to_probe


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(array, dtype=float).copy()
    result.setflags(write=False)
    return result


def _composition(values: Any, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or array.sum() <= 0:
        raise ValueError(f"{name} must be finite, non-negative, and have positive sum")
    return np.asarray(array / array.sum(), dtype=float)


def _yield_vector(
    values: Any, *, n_candidates: int, strictly_positive: bool
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != n_candidates:
        raise ValueError("effective_yield must contain one value per candidate")
    invalid_sign = array <= 0 if strictly_positive else array < 0
    if not np.all(np.isfinite(array)) or np.any(invalid_sign):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"effective_yield must contain finite {qualifier} values")
    return array


def effective_observation_yield(
    recovery_probability: Any, panel_visibility: Any
) -> NDArray[np.float64]:
    """Combine physical recovery and gate visibility into the effective yield
    (main Section 2.3).

    The effective observation yield ``e_k = r_k * v_k`` is the probability that
    one source molecule of origin ``k`` produces one accepted trace: physical
    recovery ``r_k`` times the probability ``v_k`` of passing any trace gate.
    Each input may be a scalar or a one-dimensional candidate vector; the result
    lies in ``[0, 1]`` (accepted traces per source molecule, one-trace model).

    Examples
    --------
    >>> import numpy as np
    >>> # an any-positive gate lets origin A (12 opportunities) through more
    >>> # often than origin B (3); here both are fully recovered (r = 1).
    >>> visibility = np.array([0.9313, 0.4880])
    >>> effective_observation_yield(1.0, visibility).round(4)
    array([0.9313, 0.488 ])
    >>> effective_observation_yield(0.9, 0.5)   # scalars broadcast
    array([0.45])
    """

    try:
        recovery, visibility = np.broadcast_arrays(
            np.asarray(recovery_probability, dtype=float),
            np.asarray(panel_visibility, dtype=float),
        )
    except ValueError as exc:
        raise ValueError(
            "recovery_probability and panel_visibility must be broadcast-compatible"
        ) from exc
    if recovery.ndim > 1 or recovery.size == 0:
        raise ValueError("yield inputs must be scalars or one-dimensional vectors")
    if (
        not np.all(np.isfinite(recovery))
        or not np.all(np.isfinite(visibility))
        or np.any((recovery < 0) | (recovery > 1))
        or np.any((visibility < 0) | (visibility > 1))
    ):
        raise ValueError("recovery and visibility must contain probabilities in [0, 1]")
    return _readonly(np.atleast_1d(recovery * visibility))


def source_to_analyzed_composition(
    source_composition: Any, effective_yield: Any
) -> NDArray[np.float64]:
    """Map source composition to expected composition among accepted records.

    Accepted-trace composition is ``pi_k`` proportional to ``theta_k * e_k``,
    where ``theta`` is the source composition and ``e`` the effective
    observation yield.  This is the forward selection the assay applies; its
    inverse is :func:`analyzed_to_source_composition`.

    Examples
    --------
    >>> import numpy as np
    >>> # a 50:50 source seen through yields e = (0.9313, 0.4880) tilts to A
    >>> source_to_analyzed_composition([0.5, 0.5], [0.9313, 0.4880]).round(4)
    array([0.6562, 0.3438])
    """

    source = _composition(source_composition, name="source_composition")
    exposure = _yield_vector(
        effective_yield, n_candidates=source.size, strictly_positive=False
    )
    weighted = source * exposure
    if weighted.sum() <= 0:
        raise ValueError("source composition and effective_yield produce no observations")
    return _readonly(weighted / weighted.sum())


def analyzed_to_source_composition(
    analyzed_composition: Any, effective_yield: Any
) -> NDArray[np.float64]:
    """Inverse-yield correction from accepted records back to source composition
    (main Section 2.3).

    Undoes the selection the assay applied: ``theta_k`` proportional to
    ``pi_k / e_k``, renormalized.  This is a *separate* operation from
    conditioning the trace likelihood on the gate
    (:func:`condition_on_deterministic_gate`) -- do not apply both to the same
    quantity, and do not divide by ``v_k`` twice.  Every yield must be strictly
    positive: a zero-yield source is unobservable and cannot be recovered from
    accepted traces alone.

    Examples
    --------
    >>> import numpy as np
    >>> # recover the 50:50 source from the accepted 65.6:34.4 composition
    >>> analyzed_to_source_composition([0.6562, 0.3438], [0.9313, 0.4880]).round(4)
    array([0.5, 0.5])
    """

    analyzed = _composition(analyzed_composition, name="analyzed_composition")
    exposure = _yield_vector(
        effective_yield, n_candidates=analyzed.size, strictly_positive=True
    )
    corrected = analyzed / exposure
    return _readonly(corrected / corrected.sum())


def _expanded_cycle_probabilities(
    Q: Any, cycle_to_probe: Any | None
) -> NDArray[np.float64]:
    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("Q must be a non-empty candidate-by-probe matrix")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Q must contain probabilities in [0, 1]")
    if cycle_to_probe is None:
        return q
    raw_mapping = np.asarray(cycle_to_probe)
    if raw_mapping.ndim != 1 or raw_mapping.size == 0:
        raise ValueError("cycle_to_probe must be a non-empty one-dimensional vector")
    mapping = validate_cycle_to_probe(
        raw_mapping,
        n_cycles=int(raw_mapping.size),
        n_logical_probes=int(q.shape[1]),
    )
    return np.asarray(q[:, mapping], dtype=float)


def positive_gate_visibility(
    Q: Any,
    required_groups: Sequence[Sequence[int]] | None = None,
    *,
    cycle_to_probe: Any | None = None,
    missing_rate: Any = 0.0,
) -> NDArray[np.float64]:
    """Visibility ``v_k`` under an any-positive acceptance gate, independent calls.

    ``required_groups=None`` requires at least one recorded positive across all
    physical cycles, so ``v_k = 1 - prod_c (1 - q_{k,c})``.  Otherwise each
    supplied, non-empty, non-overlapping group of zero-based physical-cycle
    indices must contain at least one recorded positive (e.g. an N-terminal and
    a C-terminal group for a both-anchors rule), and the group probabilities
    multiply.  Cycles outside the groups do not affect the gate.  A call in
    cycle ``c`` is a recorded positive with probability ``q * (1 - missing_rate)``;
    ``missing_rate`` is assumed independent of the hidden call and origin.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([
    ...     [0.2] * 12,                    # origin A: 12 binding opportunities
    ...     [0.2, 0.2, 0.2] + [0.0] * 9,   # origin B: only 3
    ... ])
    >>> positive_gate_visibility(Q).round(4)          # any positive in 12 cycles
    array([0.9313, 0.488 ])
    >>> # both-ends rule: >=1 positive in cycles 0-2 AND >=1 in cycles 3-5
    >>> positive_gate_visibility(Q, [(0, 1, 2), (3, 4, 5)]).round(4)
    array([0.2381, 0.    ])
    """

    cycle_q = _expanded_cycle_probabilities(Q, cycle_to_probe)
    n_cycles = int(cycle_q.shape[1])
    try:
        missing = np.broadcast_to(np.asarray(missing_rate, dtype=float), (n_cycles,))
    except ValueError as exc:
        raise ValueError(
            "missing_rate must be scalar or contain one value per physical cycle"
        ) from exc
    if not np.all(np.isfinite(missing)) or np.any((missing < 0) | (missing > 1)):
        raise ValueError("missing_rate must contain probabilities in [0, 1]")
    recorded_positive = cycle_q * (1.0 - missing[None, :])

    groups: Sequence[Sequence[int]]
    groups = (tuple(range(n_cycles)),) if required_groups is None else required_groups
    if len(groups) == 0:
        raise ValueError("required_groups must contain at least one cycle group")

    used: set[int] = set()
    visibility = np.ones(cycle_q.shape[0], dtype=float)
    for group in groups:
        indices = np.asarray(tuple(group))
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("every required cycle group must be non-empty")
        try:
            numeric = np.asarray(indices, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("cycle groups must contain integer indices") from exc
        if (
            not np.all(np.isfinite(numeric))
            or not np.all(numeric == np.floor(numeric))
            or np.any((numeric < 0) | (numeric >= n_cycles))
        ):
            raise ValueError("cycle groups contain an invalid physical-cycle index")
        integer = numeric.astype(int)
        if len(set(integer.tolist())) != integer.size:
            raise ValueError("a required cycle group may not repeat an index")
        overlap = used.intersection(integer.tolist())
        if overlap:
            raise ValueError("required cycle groups must not overlap")
        used.update(integer.tolist())
        visibility *= 1.0 - np.prod(1.0 - recorded_positive[:, integer], axis=1)
    return _readonly(visibility)


def condition_on_deterministic_gate(
    log_likelihoods: Any, panel_visibility: Any
) -> NDArray[np.float64]:
    """Condition pre-gate trace log-likelihoods on a deterministic gate
    (retention conditioning, main Section 2.3 / Supplementary S2.3).

    Subtracts ``log(v_k)`` so the likelihood matches the accepted population
    (``L_acc = f_k / v_k``).  Supply only *pre-gate* likelihoods, and only for
    traces that passed the same gate used to compute ``panel_visibility``.
    Never apply this to likelihoods already calibrated conditional on retention,
    and never in place of the separate inverse-yield source correction
    (:func:`analyzed_to_source_composition`) -- they solve different problems.

    Examples
    --------
    >>> import numpy as np
    >>> logs = np.array([[-1.0, -2.0]])          # one trace, two candidates
    >>> visibility = np.array([0.9313, 0.4880])
    >>> condition_on_deterministic_gate(logs, visibility).round(4)
    array([[-0.9288, -1.2826]])
    """

    logs = np.asarray(log_likelihoods, dtype=float)
    if logs.ndim != 2 or logs.shape[0] == 0 or logs.shape[1] == 0:
        raise ValueError("log_likelihoods must be a non-empty trace-by-candidate matrix")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_likelihoods may contain finite values or -inf only")
    visibility = _yield_vector(
        panel_visibility, n_candidates=logs.shape[1], strictly_positive=True
    )
    if np.any(visibility > 1):
        raise ValueError("panel_visibility must contain probabilities in (0, 1]")
    return _readonly(logs - np.log(visibility)[None, :])

