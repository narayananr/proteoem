import numpy as np
import pytest

from proteoem import (
    candidate_log_likelihoods,
    fit_likelihood_em,
    simulate_traces,
)
from proteoem.validation_oracles import (
    deterministic_profile_compatibility,
    fit_binary_compatibility_em,
    fixed_mixture_log_likelihood,
    likelihood_ratio_compatibility,
    maximize_fixed_mixture_on_simplex,
)


@pytest.mark.parametrize(
    ("profiles", "truth", "n_molecules", "seed"),
    [
        (
            np.array(
                [[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0]],
                dtype=np.int8,
            ),
            np.array([0.22, 0.47, 0.31]),
            800,
            107,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            np.array([0.07, 0.11, 0.16, 0.09, 0.13, 0.18, 0.14, 0.12]),
            6_000,
            211,
        ),
    ],
)
def test_independent_simplex_optimizer_matches_em(
    profiles, truth, n_molecules, seed
):
    simulated = simulate_traces(
        profiles,
        n_molecules,
        weights=truth,
        alpha=0.84,
        beta=0.07,
        seed=seed,
    )
    logs = candidate_log_likelihoods(simulated.observations, simulated.Q)
    em = fit_likelihood_em(logs, max_iter=20_000, tol=1e-11)
    direct = maximize_fixed_mixture_on_simplex(logs, max_iter=50_000, tol=1e-7)

    assert em.converged
    assert direct.converged
    np.testing.assert_allclose(direct.weights, em.weights, atol=2e-7, rtol=0)
    assert abs(direct.log_likelihood - em.log_likelihood) < 2e-7
    assert (
        abs(fixed_mixture_log_likelihood(logs, em.weights) - em.log_likelihood)
        < 2e-8
    )


def test_deterministic_affinity_likelihood_is_exact_binary_compatibility():
    profiles = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    observations = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [1, -1]],
        dtype=np.int8,
    )
    counts = np.array([35, 15, 20, 10, 10, 10])
    compatibility = deterministic_profile_compatibility(observations, profiles)
    deterministic_logs = candidate_log_likelihoods(observations, profiles.astype(float))

    np.testing.assert_array_equal(np.isfinite(deterministic_logs), compatibility)
    np.testing.assert_array_equal(deterministic_logs[compatibility], 0.0)

    reference = fit_binary_compatibility_em(
        compatibility, counts=counts, max_iter=100_000, tol=1e-13
    )
    em = fit_likelihood_em(
        deterministic_logs, counts=counts, max_iter=20_000, tol=1e-12
    )
    closed_form = np.array([0.42, 0.18, 0.8 / 3.0, 0.4 / 3.0])

    assert reference.converged
    assert em.converged
    np.testing.assert_allclose(reference.weights, closed_form, atol=2e-12, rtol=0)
    np.testing.assert_allclose(em.weights, closed_form, atol=2e-11, rtol=0)
    np.testing.assert_allclose(reference.weights, em.weights, atol=2e-11, rtol=0)
    assert abs(reference.log_likelihood - em.log_likelihood) < 2e-10


def test_likelihood_ratio_thresholds_have_declared_boundary_behavior():
    logs = np.log(np.array([[0.8, 0.4, 0.08], [0.2, 0.2, 0.01]]))

    top = likelihood_ratio_compatibility(logs, 1.0)
    half = likelihood_ratio_compatibility(logs, 0.5)
    positive = likelihood_ratio_compatibility(logs, 0.0)

    np.testing.assert_array_equal(top, [[True, False, False], [True, True, False]])
    np.testing.assert_array_equal(half, [[True, True, False], [True, True, False]])
    np.testing.assert_array_equal(positive, np.ones_like(logs, dtype=bool))


def test_binary_reference_rejects_an_unsupported_compatibility_row():
    with pytest.raises(ValueError, match="support"):
        fit_binary_compatibility_em([[1, 0], [0, 0]])
