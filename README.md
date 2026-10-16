![OpenOutLearn Logo](docs/logo.png)

# OpenOutLearn

A GP regression + BALD active-learning engine over text embeddings — the
scoring mechanism shared by [OpenOutFind](https://github.com/eracle/OpenOutFind)'s
lead qualifier and [OpenOutNews](https://github.com/eracle/OpenOutNews)'s
article qualifier, extracted once both had converged on the same code under
two different domain labels.

`GPBaldQualifier` fits an sklearn Pipeline (StandardScaler -> exact
GaussianProcessRegressor) on whatever labelled embeddings you give it, and
scores new candidates two ways: `predict_probs` (P(f > 0.5) from the GP
posterior, for exploitation) and `compute_bald` (information gain via MC
sampling, for exploration). `acquisition_mode`/`acquisition_scores` pick
between the two by the current label balance — exploit once real negatives
outnumber real positives, explore while the classes are still even.

`openoutlearn.calibration` turns a score into a decision. From verdicts on a
sample the model did not choose (stratified by score), `fit_rate_curve` fits the
fit rate at each score as an isotonic regression; `break_even_rate` prices the
alternative — searching for new candidates and judging those worth it — and
`lowest_bar` returns the lowest score whose fit rate beats it. Costs are in
whatever unit the caller uses; only their ratio matters. Score the sample with
`GPBaldQualifier.out_of_fold_probs` so no score comes from a fit that saw its
own label.

This is a pure library: no store, no ORM, no CLI, no persistence beyond the
fastembed model download. Each domain wires its own labels, its own store,
and its own interactive surface on top — see `openoutlearn/qualifier.py`'s
docstring for what's deliberately *not* here (cold-start anchors, class
balancing) and why those stay domain-side.

## Install

```bash
pip install openoutlearn
```

## Use

```python
import numpy as np
from pathlib import Path
from openoutlearn import GPBaldQualifier, embed_texts

cache_dir = Path.home() / ".cache" / "openoutlearn" / "fastembed"
embeddings = embed_texts(["a labelled example", "another one"], cache_dir=cache_dir)

q = GPBaldQualifier(embedding_dim=384)
q.warm_start(embeddings, np.array([1, 0]))

candidates = embed_texts(["a new candidate"], cache_dir=cache_dir)
mode, scores = q.acquisition_scores(candidates)
```

## Test

```bash
pip install -e ".[dev]"
pytest
```
