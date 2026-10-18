import numpy as np
import pytest

from openoutlearn import qualifier as qualifier_module
from openoutlearn.qualifier import GPBaldQualifier, gpr_for, record_search


def _qualifier(labels, dim=4, seed=42):
    q = GPBaldQualifier(seed=seed, embedding_dim=dim)
    X = np.array([np.full(dim, i, dtype=np.float64) for i in range(len(labels))])
    q.warm_start(X, np.array(labels))
    return q


def test_cold_start_returns_none_before_both_labels_exist():
    q = _qualifier([1, 1, 1])
    assert q.acquisition_mode() is None
    assert q.acquisition_scores(np.zeros((2, 4))) is None
    assert q.predict_probs(np.zeros((2, 4))) is None
    assert q.compute_bald(np.zeros((2, 4))) is None


def test_cold_start_returns_none_with_a_single_observation():
    q = _qualifier([1])
    assert q.acquisition_mode() is None


def test_acquisition_mode_explores_while_classes_are_balanced():
    q = _qualifier([1, 0])
    assert q.acquisition_mode() == "explore (BALD)"


def test_acquisition_mode_exploits_once_negatives_outnumber_positives():
    q = _qualifier([1, 0, 0])
    assert q.acquisition_mode() == "exploit (p)"


def test_acquisition_scores_returns_one_score_per_candidate():
    q = _qualifier([1, 0, 0])
    result = q.acquisition_scores(np.random.RandomState(0).randn(5, 4))
    assert result is not None
    mode, scores = result
    assert mode == "exploit (p)"
    assert scores.shape == (5,)


def test_predict_probs_stay_in_unit_interval():
    q = _qualifier([1, 0, 1, 0])
    probs = q.predict_probs(np.random.RandomState(0).randn(6, 4))
    assert probs is not None
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_update_appends_an_observation_and_invalidates_the_fit():
    q = _qualifier([1, 0])
    q._fit_if_needed()
    assert q._fitted is True

    q.update(np.zeros(4), 0)

    assert q._fitted is False
    assert q.n_obs == 3
    assert q.class_counts == (2, 1)


def test_warm_start_replaces_prior_observations():
    q = _qualifier([1, 0, 0])
    assert q.n_obs == 3

    q.warm_start(np.zeros((2, 4)), np.array([1, 1]))

    assert q.n_obs == 2
    assert q.class_counts == (0, 2)


# ── Kernel reuse ──

@pytest.fixture
def kernel_memory(monkeypatch):
    """A clean kernel memory, with every search counted as expensive enough to keep."""
    monkeypatch.setattr(qualifier_module, "_last_search", {})
    monkeypatch.setattr(qualifier_module, "CHEAP_SEARCH_SECONDS", 0.0)
    return qualifier_module._last_search


def test_the_first_fit_searches_the_kernel(kernel_memory):
    _, kept = gpr_for(4, 42, 10)
    assert kept is False


def test_a_fit_after_a_search_keeps_its_kernel(kernel_memory):
    _qualifier([1, 0, 0, 1]).predict_probs(np.zeros((1, 4)))
    searched = kernel_memory[4][0]

    gpr, kept = gpr_for(4, 42, 4)

    assert kept is True
    assert gpr.kernel == searched
    assert gpr.optimizer is None


def test_a_kept_kernel_still_learns_from_the_new_label(kernel_memory):
    _qualifier([1, 0, 0, 1]).predict_probs(np.zeros((1, 4)))
    before = _qualifier([1, 0, 0, 1]).predict_probs(np.full((1, 4), 4.0))
    after = _qualifier([1, 0, 0, 1, 1]).predict_probs(np.full((1, 4), 4.0))
    assert after[0] > before[0]


def test_the_kernel_is_searched_again_once_the_labels_grow_a_quarter(kernel_memory):
    _qualifier([1, 0, 0, 1]).predict_probs(np.zeros((1, 4)))
    _, kept = gpr_for(4, 42, 5)
    assert kept is False


def test_a_cheap_search_is_never_kept(kernel_memory, monkeypatch):
    monkeypatch.setattr(qualifier_module, "CHEAP_SEARCH_SECONDS", 3600.0)
    _qualifier([1, 0, 0, 1]).predict_probs(np.zeros((1, 4)))
    _, kept = gpr_for(4, 42, 4)
    assert kept is False


def test_record_search_remembers_the_fitted_kernel_and_its_size(kernel_memory):
    gpr, _ = gpr_for(4, 42, 4)
    gpr.fit(np.eye(4), np.array([1.0, 0.0, 0.0, 1.0]))

    record_search(4, gpr, 4, 12.5)

    assert kernel_memory[4] == (gpr.kernel_, 4, 12.5)
