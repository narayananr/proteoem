import numpy as np

from proteoem import (
    aggregate_traces,
    candidate_log_likelihoods,
    fit_em,
    simulate_traces,
)


def test_missing_cycles_are_marginalized_not_treated_as_negative():
    q = np.array([[0.9, 0.2], [0.1, 0.8]])
    observations = np.array([[1, -1], [-1, -1], [0, -1]], dtype=np.int8)
    likelihoods = candidate_log_likelihoods(observations, q)

    np.testing.assert_allclose(likelihoods[0], np.log([0.9, 0.1]))
    np.testing.assert_allclose(likelihoods[1], [0.0, 0.0])
    np.testing.assert_allclose(likelihoods[2], np.log([0.1, 0.9]))


def test_raw_and_trace_mask_aggregated_fits_are_identical():
    profiles = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    simulated = simulate_traces(
        profiles,
        5_000,
        weights=[0.3, 0.7],
        alpha=0.9,
        beta=0.1,
        missing_rate=0.25,
        seed=45,
    )
    aggregated = aggregate_traces(simulated.observations)

    raw_fit = fit_em(simulated.observations, profiles, alpha=0.9, beta=0.1)
    aggregated_fit = fit_em(aggregated, profiles, alpha=0.9, beta=0.1)

    np.testing.assert_allclose(raw_fit.weights, aggregated_fit.weights, atol=1e-12)
    np.testing.assert_allclose(
        raw_fit.log_likelihood_history,
        aggregated_fit.log_likelihood_history,
        atol=1e-10,
    )


def test_fully_missing_traces_carry_no_mixture_information():
    profiles = np.array([[1, 0], [0, 1]], dtype=np.int8)
    observations = np.full((20, 2), -1, dtype=np.int8)
    initial = np.array([0.25, 0.75])
    result = fit_em(observations, profiles, initial_weights=initial)

    np.testing.assert_allclose(result.weights, initial)
    np.testing.assert_allclose(result.log_likelihood, 0.0, atol=1e-12)
    assert result.diagnostics["missing_fraction"] == 1.0
