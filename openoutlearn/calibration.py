# openoutlearn/calibration.py
"""The bar a score must clear to be worth a verdict, derived from costs instead of chosen.

A qualifier's scores rank well long before they mean what they say: P(f > 0.5) of 0.7 is
not "70% of these fit". Two steps turn a score into a decision.

**1. The fit rate at each score** — ``fit_rate_curve``: an isotonic regression of
verdicts on scores, from a sample of verdicts on leads the model did *not* choose. The
only assumption is that the rate does not fall as the score rises. A sample stratified
by score needs no weights here: stratifying on the score does not bias the rate *at* a
given score.

**2. The break-even rate** — ``break_even_rate``. Each unit of work either judges the best
lead waiting (a fit with probability p, for one verdict's cost) or searches for more
(a search's cost, plus the verdicts its new leads are then worth). Judging is the better
use of the next unit while ``p / verdict_cost`` beats the best rate the search route can
reach — the marginal value theorem's rule for when to leave a patch. That best rate is a
maximum over how deep into a fresh page it pays to judge, so the threshold is consistent
with itself: a fresh page is judged by the same rule as the pool.

``lowest_bar`` then turns the break-even rate into a bar on the score.

Pure numpy and scikit-learn, like the rest of this package: drawing the sample, storing
it, measuring the costs and getting the verdicts belong to the domain. The scores fed to
``fit_rate_curve`` must be out-of-sample for their own verdicts
(``GPBaldQualifier.out_of_fold_probs``), or the curve is the model grading its own
homework.
"""
from __future__ import annotations

import numpy as np

# The scores a bar can sit at, lowest first. 0.01 apart: finer than the curve's own steps.
BAR_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def fit_rate_curve(probs: np.ndarray, labels: np.ndarray):
    """The fit rate at each score: a fitted, non-decreasing isotonic regression.

    Returns the fitted ``IsotonicRegression``; ``.predict(scores)`` reads the rate,
    clipped to the sample's score range at either end.
    """
    from sklearn.isotonic import IsotonicRegression

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError(f"probs and labels differ in shape: {probs.shape}, {labels.shape}")
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                              out_of_bounds="clip").fit(probs, labels)


def search_rate(pool_rates: np.ndarray, leads_per_search: float,
                search_cost: float, verdict_cost: float) -> float:
    """The best fits per unit cost the search route reaches.

    A search adds ``leads_per_search`` new leads whose fit rates are distributed like
    ``pool_rates``, and those worth it are then judged best first. Judging the top k of
    them yields their summed rates for ``search_cost + k * verdict_cost``; this is that
    ratio at its best k. Zero when a search adds nothing.
    """
    rates = np.sort(np.asarray(pool_rates, dtype=np.float64))[::-1]
    if leads_per_search <= 0 or len(rates) == 0:
        return 0.0
    share = leads_per_search / len(rates)
    fits = share * np.concatenate([[0.0], np.cumsum(rates)])
    verdicts = share * np.arange(len(rates) + 1)
    return float(np.max(fits / (search_cost + verdict_cost * verdicts)))


def break_even_rate(pool_rates: np.ndarray, leads_per_search: float,
                    search_cost: float, verdict_cost: float) -> float:
    """The fit rate below which a verdict is worth less than searching instead."""
    return verdict_cost * search_rate(pool_rates, leads_per_search, search_cost, verdict_cost)


def lowest_bar(curve, break_even: float, grid: np.ndarray = BAR_GRID) -> float | None:
    """The lowest score whose fit rate is above ``break_even``, or None if none is.

    Strictly above: a score the sample never saw fit is not worth a verdict even when
    searching is worth nothing either.
    """
    grid = np.asarray(grid, dtype=np.float64)
    above = curve.predict(grid) > break_even
    if not above.any():
        return None
    return float(grid[np.argmax(above)])
