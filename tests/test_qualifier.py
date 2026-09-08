import numpy as np

from openoutlearn.qualifier import GPBaldQualifier


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
