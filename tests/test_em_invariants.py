import numpy as np

from proteoem import fit_likelihood_em


def test_row_offsets_preserve_fit_and_all_normalization_invariants():
    logs = np.log(np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]]))
    counts = np.array([8, 2, 10])
    baseline = fit_likelihood_em(logs, counts=counts, tol=1e-12)
    shifted = fit_likelihood_em(
        logs + np.array([[-1e8], [1e8], [-5e7]]),
        counts=counts,
        tol=1e-12,
    )

    np.testing.assert_allclose(shifted.weights, baseline.weights, atol=2e-8)
    np.testing.assert_allclose(
        shifted.responsibilities, baseline.responsibilities, atol=2e-8
    )
    np.testing.assert_allclose(shifted.weights.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(shifted.expected_counts.sum(), counts.sum())
    np.testing.assert_allclose(shifted.responsibilities.sum(axis=1), 1.0)
    assert shifted.diagnostics["weight_sum_error"] <= 1e-14
    assert shifted.diagnostics["expected_count_total_error"] <= 1e-12


def test_extreme_erased_offsets_cannot_break_probability_invariants():
    logs = np.log(np.array([[0.9, 0.1], [0.2, 0.8]])) - 1e100
    result = fit_likelihood_em(logs, counts=[7, 3])

    # IEEE float64 cannot retain order-one differences after adding 1e100, but
    # the fitter must still return a valid probability distribution.
    np.testing.assert_allclose(result.weights.sum(), 1.0)
    np.testing.assert_allclose(result.expected_counts.sum(), 10.0)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0)


def test_all_candidate_rows_are_retained_but_do_not_slow_the_update():
    logs = np.array(
        [
            [0.0, -np.inf],
            [-np.inf, 0.0],
            [0.0, 0.0],
        ]
    )
    result = fit_likelihood_em(logs, counts=[60, 20, 10_000], tol=1e-12)

    np.testing.assert_allclose(result.weights, [0.75, 0.25], atol=1e-12)
    np.testing.assert_allclose(result.responsibilities[2], [0.75, 0.25])
    np.testing.assert_allclose(result.expected_counts, [7560.0, 2520.0])
    assert result.n_iter <= 3
    assert result.diagnostics["n_uninformative_observations"] == 10_000
    assert result.diagnostics["n_observations"] == 10_080


def test_zero_count_impossible_row_is_ignored_before_validation():
    result = fit_likelihood_em(
        np.array([[0.0, -np.inf], [-np.inf, -np.inf]]), counts=[3, 0]
    )
    np.testing.assert_allclose(result.weights, [1.0, 0.0])
    np.testing.assert_allclose(result.expected_counts, [3.0, 0.0])


def test_blocked_fit_without_returned_responsibilities_matches_full_fit():
    logs = np.log(
        np.array(
            [
                [0.8, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.3, 0.2, 0.9],
                [0.5, 0.4, 0.3],
            ]
        )
    )
    counts = [4, 7, 3, 11]
    full = fit_likelihood_em(logs, counts=counts, block_size=None, tol=1e-12)
    blocked = fit_likelihood_em(
        logs,
        counts=counts,
        block_size=2,
        return_responsibilities=False,
        tol=1e-12,
    )

    assert blocked.responsibilities is None
    np.testing.assert_allclose(blocked.weights, full.weights, atol=1e-12)
    np.testing.assert_allclose(blocked.expected_counts, full.expected_counts, atol=1e-12)
    np.testing.assert_allclose(
        blocked.centered_log_likelihood_history,
        full.centered_log_likelihood_history,
        atol=1e-12,
    )
