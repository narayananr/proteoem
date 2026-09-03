import numpy as np

from proteoem import (
    build_emission_matrix,
    fit_em,
    logsumexp,
    posterior_responsibilities,
    simulate_traces,
)


def test_em_recovers_mixture_and_has_monotone_history():
    profiles = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    truth = np.array([0.20, 0.50, 0.30])
    simulated = simulate_traces(
        profiles,
        20_000,
        weights=truth,
        alpha=0.94,
        beta=0.04,
        seed=17,
    )

    result = fit_em(
        simulated.observations,
        profiles,
        alpha=0.94,
        beta=0.04,
        tol=1e-10,
    )

    np.testing.assert_allclose(result.weights, truth, atol=0.015)
    assert result.converged
    assert result.n_iter > 0
    assert np.all(np.diff(result.log_likelihood_history) >= -1e-7)
    assert result.diagnostics["monotonic"] is True
    np.testing.assert_allclose(result.expected_counts.sum(), 20_000)


def test_full_q_overrides_alpha_beta_and_posteriors_normalize():
    profiles = np.array([[1, 0], [0, 1]], dtype=np.int8)
    q = np.array([[0.8, 0.3], [0.2, 0.7]])
    built = build_emission_matrix(profiles, alpha=0.01, beta=0.99, Q=q)
    np.testing.assert_array_equal(built, q)

    observations = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    responsibilities = posterior_responsibilities(observations, [0.4, 0.6], q)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0)
    assert responsibilities[0, 0] > responsibilities[0, 1]
    assert responsibilities[1, 1] > responsibilities[1, 0]


def test_logsumexp_is_stable_for_tiny_probabilities_and_negative_infinity():
    values = np.array([[-1_000.0, -1_001.0], [-np.inf, -np.inf]])
    observed = logsumexp(values, axis=1)
    expected_first = -1_000.0 + np.log1p(np.exp(-1.0))
    np.testing.assert_allclose(observed[0], expected_first)
    assert observed[1] == -np.inf
