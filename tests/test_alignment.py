import numpy as np

from proteoem import (
    alignment_profile_log_likelihoods,
    fit_likelihood_em,
    hard_assignment_counts,
)


def test_binary_alignment_profile_matches_fractional_em_special_case():
    profiles = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
    observations = np.array(
        [[1, -1], [1, -1], [-1, 1], [0, 1]], dtype=np.int8
    )
    logs = alignment_profile_log_likelihoods(observations, profiles, mode="exact")
    result = fit_likelihood_em(logs, tol=1e-12)

    assert np.all(np.diff(result.log_likelihood_history) >= -1e-10)
    np.testing.assert_allclose(result.weights.sum(), 1.0)
    # The fully observed last molecule is unique to candidate 3.
    assert result.weights[2] > 0


def test_best_stratum_always_has_a_candidate_and_hard_ties_can_split():
    profiles = np.array([[1, 0], [0, 1]], dtype=np.int8)
    observations = np.array([[1, 1], [-1, -1]], dtype=np.int8)
    logs = alignment_profile_log_likelihoods(observations, profiles, mode="best")

    assert np.all(np.any(np.isfinite(logs), axis=1))
    counts = hard_assignment_counts(logs, split_ties=True)
    np.testing.assert_allclose(counts, [1.0, 1.0])


def test_weighted_likelihood_em_preserves_nonbinary_evidence():
    logs = np.log(np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]]))
    counts = np.array([8, 2, 10])
    result = fit_likelihood_em(logs, counts=counts)
    assert result.weights[0] > 0.4
    assert result.weights[1] > 0.4
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0)
    assert np.all(result.responsibilities > 0.0)
    np.testing.assert_allclose(
        result.expected_counts,
        np.sum(counts[:, None] * result.responsibilities, axis=0),
    )


def test_multimapping_rows_receive_fractional_abundance_without_filtering():
    logs = np.array(
        [
            [0.0, -np.inf],  # A-only observations
            [0.0, 0.0],  # A/B multimappers
            [-np.inf, 0.0],  # B-only observations
        ]
    )
    result = fit_likelihood_em(logs, counts=[60, 20, 20], tol=1e-12)

    # The A/B multimappers begin at 0.5/0.5 under uniform initialization, then
    # borrow information from the unique rows through the updated mixture.
    np.testing.assert_allclose(result.weights, [0.75, 0.25], atol=1e-9)
    np.testing.assert_allclose(
        result.responsibilities[1], [0.75, 0.25], atol=1e-9
    )
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.expected_counts, [75.0, 25.0], atol=1e-8)
    assert result.diagnostics["n_observations"] == 100
