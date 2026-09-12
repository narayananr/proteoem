"""Core inference tools for iterative affinity proteomics."""

from .data import (
    AggregatedTraces,
    ProbeCountClasses,
    aggregate_probe_counts,
    aggregate_traces,
    compress_probe_count_classes,
    validate_cycle_to_probe,
    validate_observations,
)
from .alignment import alignment_profile_log_likelihoods, hard_assignment_counts
from .benchmark import (
    TauBenchmarkResult,
    TauLikePanel,
    abundance_metrics,
    default_tau_logical_probe_rates,
    default_tau_probe_rates,
    make_tau_like_panel,
    run_tau_like_benchmark,
    sparse_tau_weights,
)
from .em import (
    EMResult,
    LikelihoodEMResult,
    fit_em,
    fit_likelihood_em,
    logsumexp,
    posterior_responsibilities,
)
from .emissions import (
    build_emission_matrix,
    candidate_log_likelihoods,
    validate_profiles,
)
from .equivalence import (
    EquivalenceResult,
    find_equivalence_classes,
    find_observable_groups,
)
from .simulate import SimulationResult, simulate_traces
from .model_violation import (
    FixedQBootstrapResult,
    TraceLikelihoodClasses,
    bootstrap_fixed_q_classes,
    build_trace_likelihood_classes,
    emission_distribution_diagnostics,
    probability_assignment_metrics,
)
from .selection_simulate import (
    RetentionRule,
    SelectionSimulationResult,
    simulate_selected_traces,
)
from .observation_yield import (
    analyzed_to_source_composition,
    condition_on_deterministic_gate,
    effective_observation_yield,
    positive_gate_visibility,
    source_to_analyzed_composition,
)

__all__ = [
    "AggregatedTraces",
    "ProbeCountClasses",
    "EMResult",
    "LikelihoodEMResult",
    "TauBenchmarkResult",
    "TauLikePanel",
    "EquivalenceResult",
    "SimulationResult",
    "FixedQBootstrapResult",
    "TraceLikelihoodClasses",
    "SelectionSimulationResult",
    "RetentionRule",
    "aggregate_traces",
    "aggregate_probe_counts",
    "analyzed_to_source_composition",
    "abundance_metrics",
    "alignment_profile_log_likelihoods",
    "build_emission_matrix",
    "candidate_log_likelihoods",
    "condition_on_deterministic_gate",
    "default_tau_probe_rates",
    "default_tau_logical_probe_rates",
    "find_equivalence_classes",
    "find_observable_groups",
    "effective_observation_yield",
    "fit_em",
    "fit_likelihood_em",
    "hard_assignment_counts",
    "logsumexp",
    "make_tau_like_panel",
    "posterior_responsibilities",
    "positive_gate_visibility",
    "run_tau_like_benchmark",
    "simulate_traces",
    "bootstrap_fixed_q_classes",
    "build_trace_likelihood_classes",
    "emission_distribution_diagnostics",
    "probability_assignment_metrics",
    "simulate_selected_traces",
    "sparse_tau_weights",
    "source_to_analyzed_composition",
    "compress_probe_count_classes",
    "validate_cycle_to_probe",
    "validate_observations",
    "validate_profiles",
]

__version__ = "0.1.1"
