# openoutlearn/qualifier.py
"""GP Regression + BALD active learning, over plain numpy arrays.

Extracted from OpenOutFind's ``core/ml/qualifier.py`` (``BayesianQualifier``) and
OpenOutNews's ``ml/qualifier.py`` (``EngagementQualifier``), which had converged on
an identical engine under two different domain labels (an LLM's ICP-fit verdict vs.
reader engagement). This is the part that was actually shared: an sklearn Pipeline
(StandardScaler -> GaussianProcessRegressor) fitted on labelled embeddings, P(f > 0.5)
read off the exact GP posterior for exploitation, and BALD scores via MC sampling for
exploration.

Deliberately **not** here — these stay in each domain, subclassing ``GPBaldQualifier``:

- Cold-start anchors (synthetic positives for a class with zero real observations).
  OpenOutFind needs them; OpenOutNews accepts a cold start (``acquisition_mode``
  returns ``None`` until both labels exist) and falls back to newest-first instead.
- Majority-class balancing gated on a cold phase.
- Any store, ORM, LLM call, or CLI. This is a pure function of (embeddings, labels)
  in, scores out — no persistence, no I/O beyond the embedding model download.
"""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ── Kernel reuse ──
# Searching the kernel's hyperparameters (4 L-BFGS starts) is ~60 s at 2,000 labels, and
# one more label leaves the result unchanged; refitting the labels with a kept kernel is
# one factorisation, ~2 s. Replayed on a 2,000-label store, a kernel kept across 25%
# growth moved P(fit) by <0.01 on average and kept 98 of the top 100 picks. So the search
# runs again only once the labels have grown 25% since the last one, or when the last one
# was cheap (small stores, where the kernel still moves). Keyed by embedding size; one
# process judges one store.
KERNEL_REGROW = 1.25
CHEAP_SEARCH_SECONDS = 2.0
_last_search: dict[int, tuple] = {}


def gpr_for(embedding_dim: int, seed: int, n_fit: int):
    """A GPR for ``n_fit`` labels, and whether it keeps the last searched kernel.

    Kept: the last search's kernel, fixed (``optimizer=None``). Otherwise a fresh search,
    which the caller reports back with :func:`record_search` once it has fitted.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    last = _last_search.get(embedding_dim)
    if last is not None and last[2] >= CHEAP_SEARCH_SECONDS and n_fit < last[1] * KERNEL_REGROW:
        return GaussianProcessRegressor(kernel=last[0], optimizer=None, alpha=0.1), True
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=np.sqrt(embedding_dim)),
        n_restarts_optimizer=3,
        random_state=seed,
        alpha=0.1,
    ), False


def record_search(embedding_dim: int, gpr, n_fit: int, seconds: float) -> None:
    """Remember a fitted search's kernel, its label count and how long it took."""
    _last_search[embedding_dim] = (gpr.kernel_, n_fit, seconds)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def binary_entropy(p):
    """H(p) = -p log p - (1-p) log(1-p), safe for edge values."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def prob_above_half(mean, std):
    """P(f > 0.5) from the GP posterior."""
    return norm.sf(0.5, loc=mean, scale=std)


def gpr_predict(pipe, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform through all pipeline steps except the GPR, then predict with return_std."""
    from sklearn.pipeline import Pipeline

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_transformed = Pipeline(pipe.steps[:-1]).transform(X)
    return pipe.named_steps["gpr"].predict(X_transformed, return_std=True)


# ---------------------------------------------------------------------------
# GPBaldQualifier
# ---------------------------------------------------------------------------

class GPBaldQualifier:
    """Gaussian Process Regressor for active-learning selection over embeddings.

    Uses an sklearn Pipeline (StandardScaler -> GPR) as a single serializable brick.
    GPR gives an exact closed-form posterior (no Laplace approximation), which avoids
    the degenerate-0.5 problem that plagues GP classification on weakly separable
    embedding data. Probabilities are P(f > 0.5) from the GP posterior, which
    naturally incorporates uncertainty and stays in [0, 1] without clipping.

    BALD scores are computed via MC sampling from the GP posterior for the explore
    side; the balance of real observed labels decides which side is live.

    Training data is accumulated incrementally (``update``) or loaded in bulk
    (``warm_start``); the GPR is lazily re-fitted whenever a prediction is needed.
    """

    def __init__(self, seed: int = 42, embedding_dim: int = 384, n_mc_samples: int = 100):
        self.embedding_dim = embedding_dim
        self._seed = seed
        self._n_mc_samples = n_mc_samples
        self._pipeline = None
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._fitted = False
        self._rng = np.random.RandomState(seed)

    @property
    def n_obs(self) -> int:
        return len(self._y)

    @property
    def class_counts(self) -> tuple[int, int]:
        """Return (n_negatives, n_positives)."""
        n_pos = sum(self._y)
        return len(self._y) - n_pos, n_pos

    # ------------------------------------------------------------------
    # Update  (append + invalidate)
    # ------------------------------------------------------------------

    def update(self, embedding: np.ndarray, label: int):
        """Record a new labelled observation. Model is lazily re-fitted."""
        self._X.append(np.asarray(embedding, dtype=np.float64).ravel())
        self._y.append(int(label))
        self._fitted = False

    def warm_start(self, X: np.ndarray, y: np.ndarray):
        """Bulk-load labels. Replaces any prior observations."""
        self._X = [np.asarray(X[i], dtype=np.float64).ravel() for i in range(len(X))]
        self._y = [int(y[i]) for i in range(len(y))]
        self._fitted = False

    # ------------------------------------------------------------------
    # Training data + balancing  (hooks a subclass overrides)
    # ------------------------------------------------------------------

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The (X, y) the GP fits on. Overridden by subclasses that add anchors."""
        return np.array(self._X, dtype=np.float64), np.array(self._y, dtype=np.float64)

    def _maybe_balance(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Optionally subsample the majority class. No-op here; a subclass may override."""
        return X, y

    # ------------------------------------------------------------------
    # Lazy refit
    # ------------------------------------------------------------------

    def _fit_if_needed(self) -> bool:
        """Fit StandardScaler + GPR pipeline if dirty and feasible.

        Returns True once the model is usable — False when fewer than two
        observations exist or only one class has been seen (cold start).
        """
        if self._fitted:
            return True
        X_arr, y_arr = self._training_arrays()
        if len(y_arr) < 2 or len(np.unique(y_arr)) < 2:
            return False

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_fit, y_fit = self._maybe_balance(X_arr, y_arr)

        gpr, kept = gpr_for(self.embedding_dim, self._seed, len(y_fit))
        self._pipeline = Pipeline([("scaler", StandardScaler()), ("gpr", gpr)])
        started = time.monotonic()
        self._pipeline.fit(X_fit, y_fit)
        elapsed = time.monotonic() - started
        self._fitted = True
        if not kept:
            record_search(self.embedding_dim, gpr, len(y_fit), elapsed)
        logger.debug("qualifier fitted on %d observation(s) in %.2fs (%s kernel %s)", len(y_fit),
                     elapsed, "kept" if kept else "searched", gpr.kernel_)
        return True

    # ------------------------------------------------------------------
    # Prediction  (needs posterior std)
    # ------------------------------------------------------------------

    def predict_probs(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Predicted probability P(f > 0.5) for each candidate.

        Returns None when the model cannot be fitted yet.
        """
        if not self._fit_if_needed():
            return None
        mean, std = gpr_predict(self._pipeline, embeddings)
        return prob_above_half(mean, std)

    def out_of_fold_probs(self, indices, k: int = 5) -> np.ndarray | None:
        """P(f > 0.5) for each observation in ``indices``, from a fit that never saw it.

        ``indices`` point into the observations in the order they were added. They are
        split into ``k`` folds; each fold is scored by a copy fitted on every *other*
        observation, so a score never comes from a fit that saw its own label. A
        subclass's ``_training_arrays``/``_maybe_balance`` apply to each copy as usual.

        Deterministic: the folds and each copy's random state come from the seed alone, so
        the same observations always give the same scores, whatever ran before.

        Returns None when a fold's copy cannot be fitted.
        """
        indices = np.asarray(indices, dtype=np.int64)
        probs = np.empty(len(indices), dtype=np.float64)
        order = np.random.RandomState(self._seed).permutation(len(indices))
        for n, fold in enumerate(np.array_split(order, min(k, len(indices)))):
            fold_model = self._without(set(indices[fold].tolist()), seed=self._seed + n)
            fold_probs = fold_model.predict_probs(np.array([self._X[i] for i in indices[fold]]))
            if fold_probs is None:
                return None
            probs[fold] = fold_probs
        return probs

    def _without(self, held_out: set[int], seed: int) -> "GPBaldQualifier":
        """A shallow copy holding every observation but ``held_out``, unfitted, with its
        own random state — a shared one would make each fold depend on the last."""
        fold_model = copy.copy(self)
        fold_model._X = [x for i, x in enumerate(self._X) if i not in held_out]
        fold_model._y = [y for i, y in enumerate(self._y) if i not in held_out]
        fold_model._pipeline = None
        fold_model._fitted = False
        fold_model._rng = np.random.RandomState(seed)
        return fold_model

    # ------------------------------------------------------------------
    # BALD acquisition via GP posterior
    # ------------------------------------------------------------------

    def compute_bald(self, embeddings: np.ndarray) -> np.ndarray | None:
        """BALD scores for (N, embedding_dim) candidates.

        BALD = H(E[p]) - E[H(p)], MC-sampled from the exact GP posterior with a
        probit link. Higher BALD = the model disagrees with itself most = most
        informative to query.

        Returns None when the model cannot be fitted yet.
        """
        if not self._fit_if_needed():
            return None

        f_mean, f_std = gpr_predict(self._pipeline, embeddings)
        f_samples = (
            f_mean[np.newaxis, :]
            + f_std[np.newaxis, :] * self._rng.randn(self._n_mc_samples, len(f_mean))
        )
        p_samples = norm.cdf(f_samples - 0.5)
        p_pred = p_samples.mean(axis=0)
        H_pred = binary_entropy(p_pred)
        H_individual = binary_entropy(p_samples).mean(axis=0)
        return H_pred - H_individual

    # ------------------------------------------------------------------
    # Acquisition mode  (balance-driven)
    # ------------------------------------------------------------------

    def acquisition_mode(self) -> str | None:
        """The live acquisition axis: ``"exploit (p)"``, ``"explore (BALD)"``, or None.

        Exploit once real negatives outnumber real positives — the model has enough
        rejections to trust its favourites; explore while the classes are still even,
        spending picks on the candidates it's most unsure about. None on cold start
        (fewer than two observations, or only one class seen).
        """
        if not self._fit_if_needed():
            return None
        n_neg, n_pos = self.class_counts
        return "exploit (p)" if n_neg > n_pos else "explore (BALD)"

    def acquisition_scores(self, embeddings: np.ndarray) -> tuple[str, np.ndarray] | None:
        """Score candidates using the balance-driven acquisition strategy.

        Returns ``(strategy_name, scores)`` or ``None`` on cold start.
        """
        strategy = self.acquisition_mode()
        if strategy is None:
            return None
        scores = (
            self.predict_probs(embeddings) if strategy == "exploit (p)"
            else self.compute_bald(embeddings)
        )
        return strategy, scores
