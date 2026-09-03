import itertools

import numpy as np
import pytest

from proteoem import (
    aggregate_traces,
    analyzed_to_source_composition,
    candidate_log_likelihoods,
    condition_on_deterministic_gate,
    effective_observation_yield,
    fit_likelihood_em,
    positive_gate_visibility,
    simulate_traces,
    source_to_analyzed_composition,
)


def test_source_and_analyzed_composition_round_trip_and_common_scale():
    source = np.array([0.5, 0.5])
    opportunity_yield = np.array([1.0 - 0.8**12, 1.0 - 0.8**3])

    analyzed = source_to_analyzed_composition(source, opportunity_yield)
    np.testing.assert_allclose(analyzed, [0.6561, 0.3439], atol=1e-4)
    np.testing.assert_allclose(
        analyzed_to_source_composition(analyzed, opportunity_yield), source
    )
    np.testing.assert_allclose(
        source_to_analyzed_composition(source, opportunity_yield * 7.0), analyzed
    )
    np.testing.assert_allclose(
        analyzed_to_source_composition(analyzed, opportunity_yield * 7.0), source
    )


def test_effective_yield_combines_probabilities_and_zero_is_not_invertible():
    effective = effective_observation_yield(0.8, [0.5, 0.25])
    np.testing.assert_allclose(effective, [0.4, 0.2])

    analyzed = source_to_analyzed_composition([0.5, 0.5], [1.0, 0.0])
    np.testing.assert_allclose(analyzed, [1.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        analyzed_to_source_composition(analyzed, [1.0, 0.0])


def test_positive_gate_visibility_handles_any_positive_schedule_and_missingness():
    q = np.array([[0.2, 0.2], [0.2, 0.0]])
    schedule = np.array([0] * 6 + [1] * 6)

    observed = positive_gate_visibility(q, cycle_to_probe=schedule)
    expected = np.array([1.0 - 0.8**12, 1.0 - 0.8**6])
    np.testing.assert_allclose(observed, expected)

    with_missing = positive_gate_visibility(
        q, cycle_to_probe=schedule, missing_rate=0.5
    )
    np.testing.assert_allclose(with_missing, [1.0 - 0.9**12, 1.0 - 0.9**6])


def test_positive_gate_visibility_supports_tau_like_required_anchor_groups():
    cycle_q = np.array(
        [
            [0.8, 0.8, 0.8, 0.7, 0.7, 0.7],
            [0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
        ]
    )
    observed = positive_gate_visibility(
        cycle_q, required_groups=((0, 1, 2), (3, 4, 5))
    )
    expected = np.array(
        [
            (1.0 - 0.2**3) * (1.0 - 0.3**3),
            (1.0 - 0.5**3) * (1.0 - 0.6**3),
        ]
    )
    np.testing.assert_allclose(observed, expected)

    with pytest.raises(ValueError, match="must not overlap"):
        positive_gate_visibility(cycle_q, required_groups=((0, 1), (1, 2)))


def test_conditioned_any_positive_trace_distributions_normalize():
    q = np.array([[0.8, 0.3], [0.4, 0.6]])
    accepted = np.array(
        [trace for trace in itertools.product((0, 1), repeat=2) if any(trace)],
        dtype=np.int8,
    )
    base_logs = candidate_log_likelihoods(accepted, q)
    visibility = positive_gate_visibility(q)
    conditioned = condition_on_deterministic_gate(base_logs, visibility)

    np.testing.assert_allclose(np.exp(conditioned).sum(axis=0), 1.0)
    np.testing.assert_allclose(
        conditioned, base_logs - np.log(visibility)[None, :]
    )


def test_conditioned_required_group_trace_distributions_normalize():
    q = np.array([[0.8, 0.3, 0.7], [0.4, 0.6, 0.2]])
    groups = ((0, 1), (2,))
    accepted = np.array(
        [
            trace
            for trace in itertools.product((0, 1), repeat=3)
            if all(any(trace[index] for index in group) for group in groups)
        ],
        dtype=np.int8,
    )
    base_logs = candidate_log_likelihoods(accepted, q)
    visibility = positive_gate_visibility(q, required_groups=groups)
    conditioned = condition_on_deterministic_gate(base_logs, visibility)

    np.testing.assert_allclose(np.exp(conditioned).sum(axis=0), 1.0)
    np.testing.assert_allclose(
        conditioned, base_logs - np.log(visibility)[None, :]
    )


def test_selection_conditioning_preserves_source_posterior_factorization():
    q = np.array([[0.8, 0.3], [0.4, 0.6]])
    source = np.array([0.35, 0.65])
    recovery = np.array([0.75, 0.45])
    accepted = np.array(
        [trace for trace in itertools.product((0, 1), repeat=2) if any(trace)],
        dtype=np.int8,
    )

    base_logs = candidate_log_likelihoods(accepted, q)
    visibility = positive_gate_visibility(q)
    effective_yield = effective_observation_yield(recovery, visibility)
    analyzed = source_to_analyzed_composition(source, effective_yield)
    accepted_logs = condition_on_deterministic_gate(base_logs, visibility)

    # The accepted-mixture parameterization and the source parameterization
    # must give exactly the same posterior assignment probabilities:
    # pi_k f_k(y) / v_k is proportional to theta_k r_k f_k(y).
    accepted_joint = accepted_logs + np.log(analyzed)[None, :]
    source_joint = base_logs + np.log(source * recovery)[None, :]
    accepted_joint -= np.max(accepted_joint, axis=1, keepdims=True)
    source_joint -= np.max(source_joint, axis=1, keepdims=True)
    accepted_posterior = np.exp(accepted_joint)
    source_posterior = np.exp(source_joint)
    accepted_posterior /= accepted_posterior.sum(axis=1, keepdims=True)
    source_posterior /= source_posterior.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(accepted_posterior, source_posterior, atol=1e-14)


def test_selection_aware_em_recovers_source_and_unconditioned_fit_is_biased():
    source = np.array([0.5, 0.5])
    q = np.array(
        [
            [0.2] * 12,
            [0.2] * 3 + [0.0] * 9,
        ]
    )
    visibility = positive_gate_visibility(q)
    expected_analyzed = source_to_analyzed_composition(source, visibility)

    simulation = simulate_traces(
        None,
        50_000,
        weights=source,
        Q=q,
        seed=17,
    )
    accepted_mask = np.any(simulation.observations == 1, axis=1)
    accepted = aggregate_traces(simulation.observations[accepted_mask])
    base_logs = candidate_log_likelihoods(accepted, q)

    selection_aware = fit_likelihood_em(
        condition_on_deterministic_gate(base_logs, visibility),
        counts=accepted.counts,
        tol=1e-11,
        return_responsibilities=False,
    )
    recovered_source = analyzed_to_source_composition(
        selection_aware.weights, visibility
    )

    assert selection_aware.converged
    np.testing.assert_allclose(
        selection_aware.weights, expected_analyzed, atol=0.006
    )
    np.testing.assert_allclose(recovered_source, source, atol=0.006)

    # Omitting f_k(y) / v_k rewards the high-visibility origin a second time.
    naive = fit_likelihood_em(
        base_logs,
        counts=accepted.counts,
        tol=1e-11,
        return_responsibilities=False,
    )
    naive_source = analyzed_to_source_composition(naive.weights, visibility)

    assert naive.converged
    assert abs(naive_source[0] - source[0]) > 0.04
