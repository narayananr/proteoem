"""Source-population simulation with explicit recovery and trace selection.

The ordinary :func:`proteoem.simulate_traces` function generates the
composition among analyzed traces directly.  This module instead starts with
source molecules, then records physical recovery and any deterministic trace
gate separately.  It is intended for testing the observation-yield corrections
of manuscript main Section 2.3; it does not change the semantics of the ordinary
simulator.

The two selection mechanisms it records, physical recovery ``r`` and gate
visibility ``v``, are exactly the factors of the effective observation yield
``e = r * v`` of Section 2.3.  The result exposes the source composition
(``theta``), the accepted-trace composition (``pi``, as ``analyzed_composition``),
and ``effective_yield`` (``e``), so the inverse-yield correction ``theta ∝ pi / e``
can be checked against known ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .data import AggregatedTraces, aggregate_traces, validate_cycle_to_probe
from .emissions import build_emission_matrix
from .observation_yield import (
    effective_observation_yield,
    positive_gate_visibility,
    source_to_analyzed_composition,
)


RetentionRule = Literal["all", "any_positive", "required_positive_groups"]


def _readonly(array: Any, *, dtype: Any | None = None) -> NDArray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _composition(values: Any | None, n_origins: int) -> NDArray[np.float64]:
    if values is None:
        return np.full(n_origins, 1.0 / n_origins)
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.shape[0] != n_origins:
        raise ValueError(
            f"source_composition must contain one value for {n_origins} origins"
        )
    if (
        not np.all(np.isfinite(result))
        or np.any(result < 0)
        or result.sum() <= 0
    ):
        raise ValueError(
            "source_composition must be finite, non-negative, and have positive sum"
        )
    return np.asarray(result / result.sum(), dtype=float)


def _probability_vector(values: Any, length: int, *, name: str) -> NDArray[np.float64]:
    try:
        result = np.broadcast_to(np.asarray(values, dtype=float), (length,))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be scalar or contain {length} values") from exc
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return np.asarray(result, dtype=float)


def _physical_schedule(
    n_logical_probes: int, cycle_to_probe: Any | None
) -> NDArray[np.int64]:
    if cycle_to_probe is None:
        return np.arange(n_logical_probes, dtype=np.int64)
    raw = np.asarray(cycle_to_probe)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("cycle_to_probe must be a non-empty one-dimensional vector")
    return validate_cycle_to_probe(
        raw,
        n_cycles=int(raw.size),
        n_logical_probes=n_logical_probes,
    )


def _required_groups(
    retention_rule: RetentionRule,
    groups: Sequence[Sequence[int]] | None,
    *,
    n_cycles: int,
) -> tuple[tuple[int, ...], ...]:
    if retention_rule == "all":
        if groups is not None:
            raise ValueError(
                "required_positive_groups may be supplied only with "
                "retention_rule='required_positive_groups'"
            )
        return ()
    if retention_rule == "any_positive":
        if groups is not None:
            raise ValueError(
                "required_positive_groups may be supplied only with "
                "retention_rule='required_positive_groups'"
            )
        return (tuple(range(n_cycles)),)
    if retention_rule != "required_positive_groups":
        raise ValueError(
            "retention_rule must be 'all', 'any_positive', or "
            "'required_positive_groups'"
        )
    if groups is None:
        raise ValueError(
            "required_positive_groups is required when retention_rule is "
            "'required_positive_groups'"
        )

    normalized = tuple(tuple(group) for group in groups)
    # Reuse the analytical implementation as the single source of validation
    # for non-empty, disjoint, in-range physical-cycle groups.
    positive_gate_visibility(
        np.full((1, n_cycles), 0.5), required_groups=normalized
    )
    return tuple(tuple(int(index) for index in group) for group in normalized)


def _empirical_composition(
    identities: NDArray[np.int64], n_origins: int, *, name: str
) -> NDArray[np.float64]:
    if identities.size == 0:
        raise ValueError(f"{name} is undefined because no molecules were observed")
    counts = np.bincount(identities, minlength=n_origins).astype(float)
    return _readonly(counts / counts.sum(), dtype=float)


@dataclass(frozen=True)
class SelectionSimulationResult:
    """Provenance from a source-population selection simulation.

    ``potential_traces`` are the latent Bernoulli calls before missingness.
    ``observed_traces`` apply the missing marker ``-1`` to those calls for all
    simulated source molecules.  Generating these counterfactual traces even
    for unrecovered molecules makes the two independent selection mechanisms
    inspectable.  Only rows for which both ``recovery_mask`` and ``gate_mask``
    are true appear in ``observations`` and ``identities``.
    """

    observations: NDArray[np.int8]
    identities: NDArray[np.int64]
    source_identities: NDArray[np.int64]
    potential_traces: NDArray[np.int8]
    observed_traces: NDArray[np.int8]
    recovery_mask: NDArray[np.bool_]
    gate_mask: NDArray[np.bool_]
    accepted_mask: NDArray[np.bool_]
    source_composition: NDArray[np.float64]
    recovery_probability: NDArray[np.float64]
    panel_visibility: NDArray[np.float64]
    effective_yield: NDArray[np.float64]
    analyzed_composition: NDArray[np.float64]
    Q: NDArray[np.float64]
    cycle_Q: NDArray[np.float64]
    cycle_to_probe: NDArray[np.int64]
    missing_rate: NDArray[np.float64]
    retention_rule: RetentionRule
    required_positive_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        array_dtypes = {
            "observations": np.int8,
            "identities": np.int64,
            "source_identities": np.int64,
            "potential_traces": np.int8,
            "observed_traces": np.int8,
            "recovery_mask": bool,
            "gate_mask": bool,
            "accepted_mask": bool,
            "source_composition": float,
            "recovery_probability": float,
            "panel_visibility": float,
            "effective_yield": float,
            "analyzed_composition": float,
            "Q": float,
            "cycle_Q": float,
            "cycle_to_probe": np.int64,
            "missing_rate": float,
        }
        for name, dtype in array_dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))

    @property
    def n_source_molecules(self) -> int:
        return int(self.source_identities.size)

    @property
    def n_recovered(self) -> int:
        return int(self.recovery_mask.sum())

    @property
    def n_accepted(self) -> int:
        return int(self.accepted_mask.sum())

    @property
    def accepted_fraction(self) -> float:
        return self.n_accepted / self.n_source_molecules

    @property
    def aggregated(self) -> AggregatedTraces:
        if self.n_accepted == 0:
            raise ValueError("cannot aggregate because no traces were accepted")
        return aggregate_traces(self.observations)

    @property
    def empirical_source_composition(self) -> NDArray[np.float64]:
        """Realized origin frequencies among simulated source molecules."""

        return _empirical_composition(
            self.source_identities, self.source_composition.size, name="composition"
        )

    @property
    def empirical_analyzed_composition(self) -> NDArray[np.float64]:
        """Realized origin frequencies among accepted traces."""

        return _empirical_composition(
            self.identities,
            self.source_composition.size,
            name="accepted composition",
        )

    @property
    def empirical_effective_yield(self) -> NDArray[np.float64]:
        """Accepted traces per simulated source molecule for each origin.

        A value is ``nan`` when no source molecule of that origin happened to
        be sampled, because its empirical yield has no denominator.
        """

        source_counts = np.bincount(
            self.source_identities, minlength=self.source_composition.size
        ).astype(float)
        accepted_counts = np.bincount(
            self.identities, minlength=self.source_composition.size
        ).astype(float)
        result = np.full(source_counts.shape, np.nan, dtype=float)
        np.divide(
            accepted_counts,
            source_counts,
            out=result,
            where=source_counts > 0,
        )
        return _readonly(result, dtype=float)


def simulate_selected_traces(
    profiles: Any,
    n_source_molecules: int,
    *,
    source_composition: Any | None = None,
    recovery_probability: Any = 1.0,
    alpha: Any = 0.95,
    beta: Any = 0.05,
    Q: Any | None = None,
    cycle_to_probe: Any | None = None,
    missing_rate: Any = 0.0,
    retention_rule: RetentionRule = "all",
    required_positive_groups: Sequence[Sequence[int]] | None = None,
    seed: int | np.integer | None = None,
) -> SelectionSimulationResult:
    """Simulate source molecules, physical recovery, and trace retention.

    Parameters use the same emission convention as :func:`simulate_traces`.
    ``Q`` and the optional profiles are candidate-by-logical-probe matrices;
    ``cycle_to_probe`` expands them to physical cycles.  Missingness is applied
    before the gate, so only recorded positive calls count toward retention.

    ``retention_rule='all'`` keeps every independently registered, recovered
    trace, including all-negative and fully missing traces.  This is distinct
    from ``'any_positive'``.  The grouped rule requires at least one recorded
    positive in every supplied, non-overlapping group of zero-based physical
    cycle indices, as in an N-anchor plus C-anchor gate.
    """

    if (
        not isinstance(n_source_molecules, (int, np.integer))
        or isinstance(n_source_molecules, bool)
        or n_source_molecules <= 0
    ):
        raise ValueError("n_source_molecules must be a positive integer")

    q = build_emission_matrix(profiles, alpha=alpha, beta=beta, Q=Q)
    source = _composition(source_composition, q.shape[0])
    recovery = _probability_vector(
        recovery_probability, q.shape[0], name="recovery_probability"
    )
    schedule = _physical_schedule(q.shape[1], cycle_to_probe)
    cycle_q = np.asarray(q[:, schedule], dtype=float)
    missing = _probability_vector(
        missing_rate, cycle_q.shape[1], name="missing_rate"
    )
    groups = _required_groups(
        retention_rule,
        required_positive_groups,
        n_cycles=cycle_q.shape[1],
    )

    if retention_rule == "all":
        visibility = np.ones(q.shape[0], dtype=float)
    else:
        requested_groups = None if retention_rule == "any_positive" else groups
        visibility = positive_gate_visibility(
            q,
            required_groups=requested_groups,
            cycle_to_probe=schedule,
            missing_rate=missing,
        )
    exposure = effective_observation_yield(recovery, visibility)
    analyzed = source_to_analyzed_composition(source, exposure)

    rng = np.random.default_rng(seed)
    n_source = int(n_source_molecules)
    source_identities = rng.choice(q.shape[0], size=n_source, p=source)
    recovery_mask = rng.random(n_source) < recovery[source_identities]
    potential = (
        rng.random((n_source, cycle_q.shape[1])) < cycle_q[source_identities]
    ).astype(np.int8)
    observed = potential.copy()
    if np.any(missing > 0):
        missing_mask = rng.random(observed.shape) < missing[None, :]
        observed[missing_mask] = -1

    if retention_rule == "all":
        gate_mask = np.ones(n_source, dtype=bool)
    else:
        gate_mask = np.ones(n_source, dtype=bool)
        for group in groups:
            gate_mask &= np.any(observed[:, group] == 1, axis=1)
    accepted_mask = recovery_mask & gate_mask

    return SelectionSimulationResult(
        observations=observed[accepted_mask],
        identities=np.asarray(source_identities[accepted_mask], dtype=np.int64),
        source_identities=np.asarray(source_identities, dtype=np.int64),
        potential_traces=potential,
        observed_traces=observed,
        recovery_mask=recovery_mask,
        gate_mask=gate_mask,
        accepted_mask=accepted_mask,
        source_composition=source,
        recovery_probability=recovery,
        panel_visibility=visibility,
        effective_yield=exposure,
        analyzed_composition=analyzed,
        Q=q,
        cycle_Q=cycle_q,
        cycle_to_probe=schedule,
        missing_rate=missing,
        retention_rule=retention_rule,
        required_positive_groups=groups,
    )
