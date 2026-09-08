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

import logging
import time

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


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

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_fit, y_fit = self._maybe_balance(X_arr, y_arr)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=np.sqrt(self.embedding_dim)),
                n_restarts_optimizer=3,
                random_state=self._seed,
                alpha=0.1,
            )),
        ])
        started = time.monotonic()
        self._pipeline.fit(X_fit, y_fit)
        self._fitted = True
        logger.debug("qualifier fitted on %d observation(s) in %.2fs",
                     len(y_fit), time.monotonic() - started)
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
