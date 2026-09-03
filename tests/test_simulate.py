import numpy as np

from proteoem import aggregate_traces, find_equivalence_classes, simulate_traces


def test_simulation_is_seeded_and_does_not_use_global_state():
    profiles = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    first = simulate_traces(
        profiles,
        500,
        weights=[0.35, 0.65],
        alpha=0.9,
        beta=0.1,
        missing_rate=[0.0, 0.2, 0.5],
        seed=1234,
    )
    np.random.seed(999)
    second = simulate_traces(
        profiles,
        500,
        weights=[0.35, 0.65],
        alpha=0.9,
        beta=0.1,
        missing_rate=[0.0, 0.2, 0.5],
        seed=1234,
    )
    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.identities, second.identities)


def test_trace_and_mask_aggregation_preserves_multiplicity():
    observations = np.array(
        [
            [0, -1, 1],
            [0, -1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )
    aggregated = aggregate_traces(observations)

    assert aggregated.n_unique == 2
    assert aggregated.n_observations == 5
    assert sorted(aggregated.counts.tolist()) == [2, 3]
    expanded = aggregated.as_observations(expand=True)
    assert sorted(map(tuple, expanded.tolist())) == sorted(map(tuple, observations.tolist()))


def test_equivalence_classes_group_identical_candidate_rows():
    rows = np.array([[0.9, 0.1], [0.2, 0.8], [0.9, 0.1]])
    result = find_equivalence_classes(rows)
    assert result.groups == ((0, 2), (1,))
    assert not result.identifiable
