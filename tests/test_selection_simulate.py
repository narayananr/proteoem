import numpy as np
import pytest

from proteoem import positive_gate_visibility, simulate_selected_traces


def test_all_registered_traces_is_distinct_from_any_positive_gate():
    q = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    all_traces = simulate_selected_traces(
        None,
        100,
        Q=q,
        retention_rule="all",
        seed=11,
    )
    positive_only = simulate_selected_traces(
        None,
        100,
        Q=q,
        retention_rule="any_positive",
        seed=11,
    )

    assert all_traces.n_accepted == 100
    np.testing.assert_array_equal(all_traces.panel_visibility, [1.0, 1.0])
    assert 0 < positive_only.n_accepted < 100
    np.testing.assert_array_equal(positive_only.panel_visibility, [1.0, 0.0])
    assert np.all(positive_only.identities == 0)
    assert np.any(~positive_only.gate_mask)


def test_schedule_missingness_and_masks_are_recorded_consistently():
    q = np.array([[0.2, 0.8], [0.5, 0.4]])
    schedule = np.array([0, 0, 1, 1])
    missing = np.array([0.0, 0.5, 0.0, 0.25])
    result = simulate_selected_traces(
        None,
        2_000,
        source_composition=[0.4, 0.6],
        recovery_probability=[0.75, 0.5],
        Q=q,
        cycle_to_probe=schedule,
        missing_rate=missing,
        retention_rule="any_positive",
        seed=23,
    )

    expected_visibility = positive_gate_visibility(
        q, cycle_to_probe=schedule, missing_rate=missing
    )
    np.testing.assert_allclose(result.panel_visibility, expected_visibility)
    np.testing.assert_allclose(
        result.effective_yield, [0.75, 0.5] * expected_visibility
    )
    np.testing.assert_array_equal(result.cycle_Q, q[:, schedule])
    np.testing.assert_array_equal(
        result.gate_mask, np.any(result.observed_traces == 1, axis=1)
    )
    np.testing.assert_array_equal(
        result.accepted_mask, result.recovery_mask & result.gate_mask
    )
    np.testing.assert_array_equal(
        result.observations, result.observed_traces[result.accepted_mask]
    )
    np.testing.assert_array_equal(
        result.identities, result.source_identities[result.accepted_mask]
    )
    assert np.all(result.observed_traces[result.potential_traces == 0] != 1)
    assert np.all(result.observed_traces[result.potential_traces == 1] != 0)


def test_required_positive_groups_implements_tau_like_anchor_gate():
    q = np.array(
        [
            [0.8, 0.8, 0.8, 0.7, 0.7, 0.7],
            [0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
        ]
    )
    groups = ((0, 1, 2), (3, 4, 5))
    result = simulate_selected_traces(
        None,
        40_000,
        source_composition=[0.5, 0.5],
        Q=q,
        retention_rule="required_positive_groups",
        required_positive_groups=groups,
        seed=29,
    )

    expected_visibility = np.array(
        [
            (1.0 - 0.2**3) * (1.0 - 0.3**3),
            (1.0 - 0.5**3) * (1.0 - 0.6**3),
        ]
    )
    np.testing.assert_allclose(result.panel_visibility, expected_visibility)
    manual_gate = np.any(result.observed_traces[:, :3] == 1, axis=1) & np.any(
        result.observed_traces[:, 3:] == 1, axis=1
    )
    np.testing.assert_array_equal(result.gate_mask, manual_gate)
    np.testing.assert_allclose(
        result.empirical_effective_yield, expected_visibility, atol=0.01
    )
    np.testing.assert_allclose(
        result.empirical_analyzed_composition,
        result.analyzed_composition,
        atol=0.01,
    )


def test_source_recovery_and_gate_generate_theoretical_accepted_composition():
    q = np.array(
        [
            [0.2] * 12,
            [0.2] * 3 + [0.0] * 9,
        ]
    )
    result = simulate_selected_traces(
        None,
        50_000,
        source_composition=[0.5, 0.5],
        recovery_probability=[0.8, 0.6],
        Q=q,
        retention_rule="any_positive",
        seed=17,
    )

    np.testing.assert_allclose(
        result.panel_visibility, [1.0 - 0.8**12, 1.0 - 0.8**3]
    )
    np.testing.assert_allclose(
        result.empirical_source_composition, result.source_composition, atol=0.008
    )
    np.testing.assert_allclose(
        result.empirical_effective_yield, result.effective_yield, atol=0.008
    )
    np.testing.assert_allclose(
        result.empirical_analyzed_composition,
        result.analyzed_composition,
        atol=0.008,
    )


def test_selected_simulation_is_seeded_read_only_and_validates_gate_arguments():
    kwargs = dict(
        profiles=None,
        n_source_molecules=100,
        Q=np.array([[0.8, 0.2], [0.3, 0.7]]),
        retention_rule="required_positive_groups",
        required_positive_groups=((0,), (1,)),
        seed=101,
    )
    first = simulate_selected_traces(**kwargs)
    np.random.seed(999)
    second = simulate_selected_traces(**kwargs)

    np.testing.assert_array_equal(first.source_identities, second.source_identities)
    np.testing.assert_array_equal(first.observed_traces, second.observed_traces)
    np.testing.assert_array_equal(first.accepted_mask, second.accepted_mask)
    assert not first.observations.flags.writeable
    assert not first.source_composition.flags.writeable
    assert first.n_accepted == first.observations.shape[0] == first.identities.size

    with pytest.raises(ValueError, match="may be supplied only"):
        simulate_selected_traces(
            None,
            10,
            Q=np.ones((1, 2)),
            retention_rule="all",
            required_positive_groups=((0,),),
        )
    with pytest.raises(ValueError, match="is required"):
        simulate_selected_traces(
            None,
            10,
            Q=np.ones((1, 2)),
            retention_rule="required_positive_groups",
        )
    with pytest.raises(ValueError, match="must not overlap"):
        simulate_selected_traces(
            None,
            10,
            Q=np.ones((1, 2)),
            retention_rule="required_positive_groups",
            required_positive_groups=((0, 1), (1,)),
        )
