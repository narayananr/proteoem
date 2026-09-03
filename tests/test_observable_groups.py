import numpy as np

from proteoem import aggregate_probe_counts, find_observable_groups, fit_em


def test_exact_observable_groups_are_reported_and_summed():
    q = np.array(
        [
            [0.9, 0.2],
            [0.9, 0.2],
            [0.1, 0.8],
        ]
    )
    observations = np.array([[1, 0]] * 10 + [[0, 1]] * 2, dtype=np.int8)
    first = fit_em(
        observations,
        Q=q,
        initial_weights=[0.8, 0.1, 0.1],
        tol=1e-12,
    )
    second = fit_em(
        observations,
        Q=q,
        initial_weights=[0.1, 0.8, 0.1],
        tol=1e-12,
    )

    assert first.observable_groups == ((0, 1), (2,))
    np.testing.assert_allclose(
        first.observable_group_weights, second.observable_group_weights, atol=1e-12
    )
    assert not np.allclose(first.weights[:2], second.weights[:2])
    np.testing.assert_allclose(first.observable_group_weights.sum(), 1.0)
    np.testing.assert_allclose(
        first.observable_group_expected_counts.sum(), observations.shape[0]
    )
    np.testing.assert_allclose(
        first.observable_group_responsibilities.sum(axis=1), 1.0
    )
    assert first.diagnostics["n_unresolved_groups"] == 1


def test_observable_group_helper_is_exact_only():
    rows = np.array([[0.0], [0.09], [0.18]])
    result = find_observable_groups(rows)
    assert result.groups == ((0,), (1,), (2,))
    assert result.all_groups_singleton


def test_unscheduled_logical_probe_does_not_split_observable_groups():
    q = np.array(
        [
            [0.8, 0.1],
            [0.8, 0.9],
        ]
    )
    observations = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    schedule = np.array([0, 0])

    direct = fit_em(observations, Q=q, cycle_to_probe=schedule, tol=1e-12)
    sufficient = aggregate_probe_counts(
        observations,
        schedule,
        n_logical_probes=q.shape[1],
    )
    preaggregated = fit_em(sufficient, Q=q, tol=1e-12)

    assert direct.observable_groups == ((0, 1),)
    assert preaggregated.observable_groups == direct.observable_groups
    np.testing.assert_allclose(direct.observable_group_weights, [1.0])
    np.testing.assert_allclose(preaggregated.observable_group_weights, [1.0])


def test_whole_probe_dropout_does_not_split_observable_groups():
    q = np.array(
        [
            [0.8, 0.1],
            [0.8, 0.9],
        ]
    )
    observations = np.array(
        [
            [1, -1, 0, -1],
            [0, -1, 1, -1],
            [1, -1, 1, -1],
        ],
        dtype=np.int8,
    )
    schedule = np.array([0, 1, 0, 1])

    result = fit_em(observations, Q=q, cycle_to_probe=schedule, tol=1e-12)

    assert result.observable_groups == ((0, 1),)
    assert result.diagnostics["n_unresolved_groups"] == 1
