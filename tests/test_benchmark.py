import numpy as np

from proteoem import (
    default_tau_probe_rates,
    make_tau_like_panel,
    run_tau_like_benchmark,
    sparse_tau_weights,
)


def test_tau_like_panel_has_expected_shape_and_unique_candidate_profiles():
    panel = make_tau_like_panel(repeats=3)
    assert panel.profiles.shape == (768, 36)
    assert len(panel.candidate_ids) == 768
    assert len(set(panel.candidate_ids)) == 768
    assert np.unique(panel.profiles, axis=0).shape[0] == 768

    alpha, beta = default_tau_probe_rates(panel)
    assert alpha.shape == beta.shape == (36,)
    assert np.all(alpha > beta)


def test_sparse_tau_weights_are_seeded_and_cover_all_isoforms():
    panel = make_tau_like_panel(repeats=1)
    first = sparse_tau_weights(panel, n_active=12, seed=4)
    second = sparse_tau_weights(panel, n_active=12, seed=4)
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) == 12
    assert np.isclose(first.sum(), 1.0)
    represented = {
        isoform for isoform, weight in zip(panel.isoforms, first) if weight > 0
    }
    assert represented == set(("0N3R", "0N4R", "1N3R", "1N4R", "2N3R", "2N4R"))


def test_nonconverged_benchmark_does_not_report_comparative_metrics():
    result = run_tau_like_benchmark(n_molecules=100, max_iter=1)
    assert not result.metrics_valid
    assert result.metrics == {}
