"""Synthetic benchmark for the tau-like panel (manuscript Section 3).

These functions build the structure of a targeted iterative-affinity assay and
run the comparison reported in the paper. They model the design only; they do
not reproduce any commercial platform's chemistry, software, calibration, or raw
data.

* :func:`make_tau_like_panel` builds the candidate dictionary of Section 3.1:
  six tau isoforms times seven binary PTM sites (768 proteoforms), read by twelve
  logical probes (two pan-tau, three isoform, seven PTM), each repeated
  ``repeats`` times.
* :func:`sparse_tau_weights` draws the sparse ground-truth composition (every
  isoform represented, the rest filled to ``n_active`` active states).
* :func:`run_tau_like_benchmark` fits weighted-affinity EM (the full graded
  likelihood, main Section 2.4) and contrasts it with the two reduced baselines
  of Supplementary Methods S2.5 -- binary-incidence EM and hard top-likelihood
  counting -- on the same simulated traces (Section 3.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .alignment import alignment_profile_log_likelihoods, hard_assignment_counts
from .em import EMResult, LikelihoodEMResult, fit_em, fit_likelihood_em
from .emissions import candidate_log_likelihoods
from .simulate import SimulationResult, simulate_traces


ISOFORM_LABELS = (
    "0N3R",
    "0N4R",
    "1N3R",
    "1N4R",
    "2N3R",
    "2N4R",
)
PTM_LABELS = (
    "pT181",
    "pS202_pT205",
    "pT205",
    "pS214",
    "pT217",
    "pT231",
    "pS396",
)
BASE_PROBE_LABELS = (
    "pan_tau_A",
    "pan_tau_B",
    "anti_0N",
    "anti_2N",
    "anti_4R",
    "anti_pT181",
    "anti_pS202_pT205",
    "anti_pT205",
    "anti_pS214",
    "anti_pT217",
    "anti_pT231",
    "anti_pS396",
)


@dataclass(frozen=True)
class TauLikePanel:
    """Candidate dictionary and repeated binary probe design."""

    profiles: NDArray[np.int8]
    candidate_ids: tuple[str, ...]
    isoforms: tuple[str, ...]
    probe_ids: tuple[str, ...]
    repeats: int

    def __post_init__(self) -> None:
        profiles = np.asarray(self.profiles, dtype=np.int8).copy()
        if profiles.shape != (len(self.candidate_ids), len(self.probe_ids)):
            raise ValueError("panel metadata and profile matrix have inconsistent shapes")
        profiles.setflags(write=False)
        object.__setattr__(self, "profiles", profiles)

    @property
    def n_candidates(self) -> int:
        return int(self.profiles.shape[0])

    @property
    def n_probes(self) -> int:
        return int(self.profiles.shape[1])

    @property
    def n_logical_probes(self) -> int:
        return self.n_probes // self.repeats

    @property
    def logical_profiles(self) -> NDArray[np.int8]:
        profiles = self.profiles[:, : self.n_logical_probes].copy()
        profiles.setflags(write=False)
        return profiles

    @property
    def cycle_to_probe(self) -> NDArray[np.int64]:
        mapping = np.tile(np.arange(self.n_logical_probes), self.repeats)
        mapping.setflags(write=False)
        return mapping

    @property
    def logical_probe_ids(self) -> tuple[str, ...]:
        return tuple(
            probe_id.rsplit(":r", maxsplit=1)[0]
            for probe_id in self.probe_ids[: self.n_logical_probes]
        )


def make_tau_like_panel(*, repeats: int = 3) -> TauLikePanel:
    """Construct six isoforms x seven binary PTMs (768 candidates).

    The twelve logical probes comprise two pan-target probes, three isoform
    probes, and seven PTM probes.  Every logical probe is repeated ``repeats``
    times in pass order.
    """

    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    rows: list[list[int]] = []
    candidate_ids: list[str] = []
    candidate_isoforms: list[str] = []
    for isoform in ISOFORM_LABELS:
        n_insert = isoform[:2]
        repeat_domain = isoform[2:]
        # Isoform-level probe truth: the two pan-tau probes always bind, then the
        # N-terminal (0N / 2N) and repeat-domain (4R) isoform probes resolve the
        # backbone (the binary isoform-probe codes of Supplementary Table S3).
        isoform_bits = [
            1,
            1,
            int(n_insert == "0N"),
            int(n_insert == "2N"),
            int(repeat_domain == "4R"),
        ]
        for state in range(2 ** len(PTM_LABELS)):
            ptm_bits = [
                (state >> (len(PTM_LABELS) - 1 - bit)) & 1
                for bit in range(len(PTM_LABELS))
            ]
            rows.append(isoform_bits + ptm_bits)
            active = [name for name, value in zip(PTM_LABELS, ptm_bits) if value]
            suffix = "+".join(active) if active else "unmodified"
            candidate_ids.append(f"{isoform}|{suffix}")
            candidate_isoforms.append(isoform)

    base = np.asarray(rows, dtype=np.int8)
    profiles = np.tile(base, (1, int(repeats)))
    probe_ids = tuple(
        f"{probe}:r{repeat + 1}"
        for repeat in range(int(repeats))
        for probe in BASE_PROBE_LABELS
    )
    return TauLikePanel(
        profiles=profiles,
        candidate_ids=tuple(candidate_ids),
        isoforms=tuple(candidate_isoforms),
        probe_ids=probe_ids,
        repeats=int(repeats),
    )


def default_tau_probe_rates(
    panel: TauLikePanel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return explicit synthetic on/off-target rates for the panel."""

    base_alpha = np.asarray([0.97, 0.97] + [0.91] * 3 + [0.84] * 7)
    base_beta = np.asarray([0.002, 0.002] + [0.010] * 3 + [0.015] * 7)
    alpha = np.tile(base_alpha, panel.repeats)
    beta = np.tile(base_beta, panel.repeats)
    alpha.setflags(write=False)
    beta.setflags(write=False)
    return alpha, beta


def default_tau_logical_probe_rates(
    panel: TauLikePanel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return one synthetic on/off-target rate per logical probe."""

    alpha, beta = default_tau_probe_rates(panel)
    logical_alpha = alpha[: panel.n_logical_probes].copy()
    logical_beta = beta[: panel.n_logical_probes].copy()
    logical_alpha.setflags(write=False)
    logical_beta.setflags(write=False)
    return logical_alpha, logical_beta


def sparse_tau_weights(
    panel: TauLikePanel,
    *,
    n_active: int = 32,
    concentration: float = 0.4,
    seed: int | None = 1,
) -> NDArray[np.float64]:
    """Create a seeded sparse abundance vector with every isoform represented.

    ``concentration`` is the symmetric Dirichlet concentration on the active
    states: values below one give a heavy-tailed composition (a few dominant
    states, many rare), larger values a flatter one. The default 0.4 matches the
    frozen benchmark, so the default draw is unchanged.
    """

    if not isinstance(n_active, (int, np.integer)) or not (
        len(ISOFORM_LABELS) <= n_active <= panel.n_candidates
    ):
        raise ValueError(
            f"n_active must be between {len(ISOFORM_LABELS)} and {panel.n_candidates}"
        )
    if (
        isinstance(concentration, bool)
        or not isinstance(concentration, (int, float, np.floating, np.integer))
        or float(concentration) <= 0.0
    ):
        raise ValueError("concentration must be a positive number")
    rng = np.random.default_rng(seed)
    isoform_array = np.asarray(panel.isoforms)
    mandatory = [
        int(rng.choice(np.flatnonzero(isoform_array == isoform)))
        for isoform in ISOFORM_LABELS
    ]
    remaining_pool = np.setdiff1d(
        np.arange(panel.n_candidates), np.asarray(mandatory), assume_unique=False
    )
    extra = rng.choice(
        remaining_pool,
        size=int(n_active) - len(mandatory),
        replace=False,
    )
    active = np.concatenate((np.asarray(mandatory), np.asarray(extra, dtype=int)))
    weights = np.zeros(panel.n_candidates, dtype=float)
    draws = rng.gamma(shape=float(concentration), scale=1.0, size=active.size)
    weights[active] = draws / draws.sum()
    weights.setflags(write=False)
    return weights


def abundance_metrics(
    estimate: NDArray[np.float64], truth: NDArray[np.float64]
) -> dict[str, float]:
    """Calculate compact abundance-recovery metrics."""

    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or estimate.ndim != 1:
        raise ValueError("estimate and truth must be equal-length vectors")
    difference = estimate - truth
    return {
        "l1": float(np.sum(np.abs(difference))),
        "total_variation": float(0.5 * np.sum(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_error": float(np.max(np.abs(difference))),
    }


@dataclass(frozen=True)
class TauBenchmarkResult:
    panel: TauLikePanel
    simulation: SimulationResult
    weighted_fit: EMResult
    incidence_fit: LikelihoodEMResult
    hard_weights: NDArray[np.float64]
    metrics: dict[str, dict[str, float]]
    metrics_valid: bool


def run_tau_like_benchmark(
    *,
    n_molecules: int = 5_000,
    n_active: int = 32,
    concentration: float = 0.4,
    repeats: int = 3,
    missing_rate: float = 0.02,
    seed: int = 7,
    max_iter: int = 300,
    tol: float = 1e-7,
    block_size: int | None = 4096,
    return_responsibilities: bool = False,
) -> TauBenchmarkResult:
    """Run one reproducible oracle-calibrated synthetic benchmark.

    ``concentration`` sets the symmetric Dirichlet concentration on the active
    states (see :func:`sparse_tau_weights`) and ``repeats`` sets the number of
    physical passes per logical probe (cycles ``C = 12 * repeats``). The
    defaults reproduce the frozen benchmark exactly.
    """

    panel = make_tau_like_panel(repeats=repeats)
    alpha, beta = default_tau_probe_rates(panel)
    logical_alpha, logical_beta = default_tau_logical_probe_rates(panel)
    truth = sparse_tau_weights(
        panel, n_active=n_active, concentration=concentration, seed=seed
    )
    simulation = simulate_traces(
        panel.profiles,
        n_molecules,
        weights=truth,
        alpha=alpha,
        beta=beta,
        missing_rate=missing_rate,
        seed=seed + 1,
    )
    aggregated = simulation.aggregated

    # The method: weighted-affinity EM on the full graded likelihood (Section 2.4).
    weighted_fit = fit_em(
        simulation.observations,
        panel.logical_profiles,
        alpha=logical_alpha,
        beta=logical_beta,
        cycle_to_probe=panel.cycle_to_probe,
        max_iter=max_iter,
        tol=tol,
        block_size=block_size,
        return_responsibilities=return_responsibilities,
    )
    weighted_logs = candidate_log_likelihoods(
        weighted_fit.aggregated, weighted_fit.Q
    )
    # Reduced baseline 1 (S2.5): hard top-likelihood counting -- assign each trace
    # to its single most likely origin and count, discarding the graded evidence.
    hard_counts = hard_assignment_counts(
        weighted_logs, counts=weighted_fit.aggregated.counts, split_ties=True
    )
    hard_weights = hard_counts / hard_counts.sum()
    hard_weights.setflags(write=False)

    # Reduced baseline 2 (S2.5): binary-incidence EM -- round the likelihood to
    # 0/1 compatibility, then run the same mixture EM on that hard-set profile.
    incidence_logs = alignment_profile_log_likelihoods(
        aggregated, panel.profiles, mode="best"
    )
    incidence_fit = fit_likelihood_em(
        incidence_logs,
        counts=aggregated.counts,
        max_iter=max_iter,
        tol=tol,
        block_size=block_size,
        return_responsibilities=return_responsibilities,
    )

    metrics_valid = bool(weighted_fit.converged and incidence_fit.converged)
    metrics = (
        {
            "hard": abundance_metrics(hard_weights, truth),
            "binary_incidence_em": abundance_metrics(incidence_fit.weights, truth),
            "weighted_affinity_em": abundance_metrics(weighted_fit.weights, truth),
        }
        if metrics_valid
        else {}
    )
    return TauBenchmarkResult(
        panel=panel,
        simulation=simulation,
        weighted_fit=weighted_fit,
        incidence_fit=incidence_fit,
        hard_weights=hard_weights,
        metrics=metrics,
        metrics_valid=metrics_valid,
    )
