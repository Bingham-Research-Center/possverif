# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Python package implementing the possibilistic forecast verification scorecard from:
"Possible, Yes; Ignorant, Perhaps: A Scorecard for Possibilistic Forecasts" (Lawson et al., in prep for JAS).

The companion LaTeX manuscript lives at `../possibility-verif`. The equations and notation in this code follow that paper exactly.

## Build and Install

```bash
pip install -e .          # editable install
pip install -e ".[dev]"   # with pytest
```

## Running Tests

```bash
pytest
```

## Architecture

Single-module package for now:

- `src/possverif/distribution.py` -- all logic: `PossibilisticDistribution` class + `Scorecard` named tuple
- `src/possverif/__init__.py` -- re-exports the two public names

The `PossibilisticDistribution` class holds a raw possibility distribution (dict of category→float) and provides:
- Pre-event metrics: `commitment`, `ignorance`, `diffuseness()`, `entropy()`, `necessity()`, `conditional_necessity()`
- Post-event scorecard: `scorecard(observed)` returns a `Scorecard` named tuple with the five-number verification metrics
- Pignistic bridge: `to_probability()`, `surprise(observed)`

## Notation (must match the paper)

| Code | Paper | Definition |
|------|-------|------------|
| `raw` | π | Raw possibility distribution |
| `normalized` | π' | π/m, max-normalized shape |
| `commitment` | m | max(π) |
| `ignorance` | H_Π | 1 - m |
| `depth_of_truth` | α* | π'(c_obs) |
| `diffuseness` | η | (1/K) Σ π'(c) |
| `support_margin` | δ | α* - η |
| `conditional_necessity` | N_c* | 1 - max_{ω≠obs} π'(ω) |

## Companion Repos (do not wire paths between repos)

| Repo | Path | Role |
|------|------|------|
| possibility-verif | `../possibility-verif` | LaTeX manuscript (equations are authoritative) |
| python-nsf-dprog | `../python-nsf-dprog` | NSF proposal code (imports this package) |
| latex-nsf-dprog | `../latex-nsf-dprog` | NSF proposal narrative |
