import numpy as np
import pytest

from openoutlearn.calibration import (
    break_even_rate,
    fit_rate_curve,
    lowest_bar,
    search_rate,
)
from openoutlearn.qualifier import GPBaldQualifier


def _stratified_sample(fit_rate, n_per_band=40, seed=0):
    """Verdicts drawn evenly across ten score bands, fitting at ``fit_rate(score)``."""
    rng = np.random.RandomState(seed)
    probs = np.concatenate([rng.uniform(b / 10, (b + 1) / 10, n_per_band) for b in range(10)])
    labels = (rng.rand(len(probs)) < fit_rate(probs)).astype(int)
    return probs, labels


# ── The fit-rate curve ──

def test_the_curve_recovers_a_known_fit_rate():
    probs, labels = _stratified_sample(lambda p: p, n_per_band=400)

    curve = fit_rate_curve(probs, labels)

    for score in (0.15, 0.45, 0.85):
        assert curve.predict([score])[0] == pytest.approx(score, abs=0.08)


def test_the_curve_never_falls_as_the_score_rises():
    probs, labels = _stratified_sample(lambda p: p ** 2, n_per_band=20)

    rates = fit_rate_curve(probs, labels).predict(np.linspace(0, 1, 101))

    assert np.all(np.diff(rates) >= 0)


def test_a_sample_with_no_fits_has_a_flat_zero_curve():
    probs, labels = _stratified_sample(lambda p: np.zeros_like(p))

    assert np.all(fit_rate_curve(probs, labels).predict(np.linspace(0, 1, 11)) == 0)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        fit_rate_curve(np.zeros(3), np.zeros(2))


# ── The search route and the break-even ──

def test_search_rate_judges_only_the_new_leads_worth_it():
    # Ten leads a search, half at 0.5 and half at 0. Judging the good half yields 2.5 fits
    # for 100 + 5×10 = 150 seconds; judging all ten would add cost and no fits.
    rates = np.array([0.5] * 5 + [0.0] * 5)

    rate = search_rate(rates, leads_per_search=10, search_cost=100, verdict_cost=10)

    assert rate == pytest.approx(2.5 / 150)


def test_a_search_that_adds_nothing_is_worth_nothing():
    assert search_rate(np.array([0.9, 0.8]), leads_per_search=0,
                       search_cost=70, verdict_cost=5) == 0.0


def test_cheaper_verdicts_lower_the_break_even():
    rates = np.linspace(0, 0.8, 1000)

    cheap = break_even_rate(rates, leads_per_search=10, search_cost=70, verdict_cost=5)
    dear = break_even_rate(rates, leads_per_search=10, search_cost=70, verdict_cost=60)

    assert 0 < cheap < dear < 0.8


def test_the_break_even_is_self_consistent():
    """At the break-even, judging a pool lead and judging the marginal new lead of the best
    search plan pay the same: the threshold is the one the search route itself uses."""
    rates = np.linspace(0, 0.8, 1000)
    leads, search, verdict = 10, 70, 5

    c = break_even_rate(rates, leads, search, verdict)
    # Judge every new lead above c, none below: that plan must reach the search rate.
    worth = rates[rates > c]
    plan_rate = (leads / len(rates)) * worth.sum() / (search + verdict * leads * len(worth) / len(rates))

    assert plan_rate == pytest.approx(search_rate(rates, leads, search, verdict), rel=0.01)


# ── The bar ──

def test_the_bar_is_the_lowest_score_above_the_break_even():
    probs, labels = _stratified_sample(lambda p: p, n_per_band=400)
    curve = fit_rate_curve(probs, labels)

    bar = lowest_bar(curve, break_even=0.3)

    assert 0.25 <= bar <= 0.4
    assert curve.predict([bar])[0] > 0.3


def test_no_score_above_the_break_even_has_no_bar():
    probs, labels = _stratified_sample(lambda p: 0.1 * np.ones_like(p), n_per_band=200)

    assert lowest_bar(fit_rate_curve(probs, labels), break_even=0.5) is None


def test_a_zero_break_even_still_skips_scores_that_never_fit():
    probs, labels = _stratified_sample(lambda p: (p > 0.5).astype(float), n_per_band=50)

    assert lowest_bar(fit_rate_curve(probs, labels), break_even=0.0) >= 0.4


# ── Out-of-fold scores ──

class _RecordingQualifier(GPBaldQualifier):
    """Remembers every training set a fit was handed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fits: list[np.ndarray] = []

    def _training_arrays(self):
        X, y = super()._training_arrays()
        self.fits.append(X)
        return X, y


class _SubsamplingQualifier(GPBaldQualifier):
    """Drops a random tenth of its training set, as a balancing subclass would."""

    def _maybe_balance(self, X, y):
        keep = self._rng.rand(len(y)) > 0.1
        return X[keep], y[keep]


def _labelled(n=20, dim=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_an_out_of_fold_score_never_comes_from_a_fit_that_saw_its_label():
    X, y = _labelled()
    qualifier = _RecordingQualifier(embedding_dim=4)
    qualifier.warm_start(X, y)
    held_out = np.arange(0, 20, 2)

    probs = qualifier.out_of_fold_probs(held_out, k=5)

    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert len(qualifier.fits) == 5
    for fit in qualifier.fits:
        assert len(fit) == 18
    for i in held_out:
        # Seen by exactly four of the five fits: every fold but its own.
        seen = sum(any(np.array_equal(X[i], row) for row in fit) for fit in qualifier.fits)
        assert seen == 4


def test_out_of_fold_scores_are_the_same_whatever_ran_before():
    X, y = _labelled(n=30)
    first = _SubsamplingQualifier(embedding_dim=4)
    first.warm_start(X, y)
    second = _SubsamplingQualifier(embedding_dim=4)
    second.warm_start(X, y)
    second._rng.rand(1000)  # a history the first never had

    np.testing.assert_array_equal(first.out_of_fold_probs(np.arange(10), k=5),
                                  second.out_of_fold_probs(np.arange(10), k=5))


def test_out_of_fold_scores_leave_the_model_itself_untouched():
    X, y = _labelled()
    qualifier = GPBaldQualifier(embedding_dim=4)
    qualifier.warm_start(X, y)

    qualifier.out_of_fold_probs(np.arange(5), k=5)

    assert qualifier.n_obs == 20
    assert qualifier._fitted is False


def test_out_of_fold_scores_are_none_when_a_fold_cannot_fit():
    qualifier = GPBaldQualifier(embedding_dim=4)
    qualifier.warm_start(np.random.RandomState(0).randn(3, 4), np.array([1, 0, 0]))

    # Holding out the only positive leaves a one-class fit.
    assert qualifier.out_of_fold_probs(np.array([0]), k=1) is None
