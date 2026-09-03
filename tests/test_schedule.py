import numpy as np

from proteoem import (
    aggregate_probe_counts,
    build_emission_matrix,
    candidate_log_likelihoods,
    compress_probe_count_classes,
    fit_em,
)


def test_exchangeable_repeats_use_positive_and_observed_counts():
    observations = np.array(
        [
            [1, 0, -1, 0, 0, -1],
            [0, 0, 1, -1, -1, 0],
            [1, 1, -1, 0, 0, -1],
        ],
        dtype=np.int8,
    )
    schedule = np.array([0, 1, 0, 1, 0, 1])
    classes = aggregate_probe_counts(observations, schedule)

    # The first two ordered traces have the same positive/observed counts for
    # each logical probe; the third differs for probe 1.
    assert classes.n_unique == 2
    assert classes.n_observations == 3
    assert sorted(classes.counts.tolist()) == [1, 2]
    assert classes.n_cycles == 6
    assert classes.n_logical_probes == 2


def test_scheduled_fit_matches_manual_cycle_expansion_and_raw_kernel():
    logical_profiles = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    schedule = np.array([0, 1, 0, 1, 0, 1])
    logical_q = build_emission_matrix(
        logical_profiles,
        alpha=[0.88, 0.81],
        beta=[0.07, 0.12],
    )
    observations = np.array(
        [
            [1, 0, 0, 0, -1, 0],
            [0, 1, 0, 1, 0, -1],
            [1, 1, 1, 0, 1, 1],
            [0, -1, 1, 0, 0, 0],
            [1, 0, 0, 0, -1, 0],
        ],
        dtype=np.int8,
    )

    scheduled = fit_em(
        observations,
        Q=logical_q,
        cycle_to_probe=schedule,
        tol=1e-12,
    )
    expanded = fit_em(observations, Q=logical_q[:, schedule], tol=1e-12)
    molecule_logs = candidate_log_likelihoods(observations, logical_q[:, schedule])
    from proteoem import fit_likelihood_em

    molecule_fit = fit_likelihood_em(molecule_logs, tol=1e-12)

    np.testing.assert_allclose(scheduled.weights, expanded.weights, atol=3e-12)
    np.testing.assert_allclose(scheduled.weights, molecule_fit.weights, atol=1e-12)
    np.testing.assert_allclose(
        scheduled.log_likelihood, expanded.log_likelihood, atol=1e-11
    )
    np.testing.assert_allclose(
        scheduled.log_likelihood, molecule_fit.log_likelihood, atol=1e-11
    )
    assert scheduled.diagnostics["n_sufficient_count_classes"] <= observations.shape[0]
    assert scheduled.diagnostics["n_trace_likelihood_classes"] <= scheduled.diagnostics[
        "n_sufficient_count_classes"
    ]


def test_origin_invariant_probe_changes_offset_not_assignments():
    q = np.array(
        [
            [0.9, 0.8],
            [0.9, 0.2],
        ]
    )
    schedule = np.array([0, 1, 0, 1])
    observations = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    sufficient = aggregate_probe_counts(observations, schedule)
    relative = compress_probe_count_classes(sufficient, q)

    assert sufficient.n_unique == 3
    assert relative.n_unique == 2
    np.testing.assert_array_equal(relative.probe_indices, [1])
    assert relative.log_likelihood_offset < 0

    scheduled = fit_em(observations, Q=q, cycle_to_probe=schedule, tol=1e-12)
    expanded = fit_em(observations, Q=q[:, schedule], tol=1e-12)
    np.testing.assert_allclose(scheduled.weights, expanded.weights, atol=3e-12)
    np.testing.assert_allclose(
        scheduled.log_likelihood, expanded.log_likelihood, atol=1e-12
    )


def test_invalid_cycle_schedule_is_rejected():
    observations = np.array([[1, 0, 1]], dtype=np.int8)
    q = np.array([[0.9, 0.2], [0.1, 0.8]])
    for schedule in ([0, 1], [0, 1, 2], [0, -1, 1], [0, 0.5, 1]):
        try:
            fit_em(observations, Q=q, cycle_to_probe=schedule)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid schedule was accepted: {schedule}")
