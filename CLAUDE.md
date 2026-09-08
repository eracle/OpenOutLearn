# CLAUDE.md

## Rules

- **Commits**: No `Co-Authored-By` lines. Single-line messages (no body). Same convention as
  OpenOutFind and OpenOutNews.
- **Always work on `main`.** Never create or switch to a feature branch — commit directly to
  `main`.
- **Every green push to `main` is a release.** `.github/workflows/deploy.yml` runs the test job,
  then derives the version from the commit count and publishes to PyPI via trusted publishing
  (environment `pypi`) — same pattern as OpenOutFind, OpenOutSend and OpenOutNews. `needs: test`
  is the whole gate; there is no manual release step.
- **No memory**: never use the auto-memory system (no MEMORY.md, no memory files).
- **No CLI, no store, on purpose.** See *Project Overview* below for why — do not add one back in
  without re-reading that reasoning.

## Project Overview

`openoutlearn` is a GP regression + BALD active-learning engine over text embeddings —
`GPBaldQualifier` (`openoutlearn/qualifier.py`) plus `embed_text`/`embed_texts`
(`openoutlearn/embeddings.py`, a fastembed wrapper). It's the scoring mechanism shared by
[OpenOutFind](https://github.com/eracle/OpenOutFind)'s lead qualifier (`BayesianQualifier`) and
[OpenOutNews](https://github.com/eracle/OpenOutNews)'s article qualifier
(`EngagementQualifier`) — extracted once both repos had converged on identical code under two
different domain labels (an LLM's ICP-fit verdict vs. reader engagement). See
`roadmap/history/2026-09-08-p2-e3-openoutnews-first-version.md` in `openoutreach-docs` for how
that extraction happened.

`GPBaldQualifier` fits an sklearn Pipeline (StandardScaler -> exact GaussianProcessRegressor) on
whatever labelled embeddings a caller gives it, and scores new candidates two ways:
`predict_probs` (P(f > 0.5) from the GP posterior, for exploitation) and `compute_bald`
(information gain via MC sampling, for exploration). `acquisition_mode`/`acquisition_scores`
switch between the two by the current label balance.

**This is deliberately a plain library — no store, no ORM, no CLI, no persistence beyond the
fastembed model download.** An interactive "pass text, see a score" loop needs state across
calls (every embedding+label pair labelled so far), which would mean this repo growing its own
mini version of what `Lead`/`Candidate` already are in each domain's own store — duplicating real
domain modelling for no real consumer. The interactive/agentic surface stays where the real data
already lives: OpenOutFind's `find-leads` skill and OpenOutNews's own CLI, each built on top of
this engine. Do not add a CLI, a database, or a "debug/tune the model standalone" command to this
repo without a real consumer that needs it — see the `openoutreach-docs` conversation that decided
this (search its session history for "openoutlearn" if picking this back up).

**Subclasses add what's domain-specific; this repo stays generic.** `BayesianQualifier`
(OpenOutFind) adds cold-start anchors and majority-class balancing gated on a cold phase.
`EngagementQualifier` (OpenOutNews) needs neither — it accepts a plain cold start
(`acquisition_mode` returns `None` until both a positive and a negative label exist) and falls
back to newest-first. Anything that only one domain needs belongs in that domain's subclass, not
here.

## Commands

```bash
pip install -e ".[dev]"
pytest
```

## Publishing

`pyproject.toml`'s `version` is the *base* (major.minor, bumped by hand); the patch is the commit
count at publish time, same derivation as the other three repos. There is no release gate beyond
`needs: test` — every green push to `main` ships.
