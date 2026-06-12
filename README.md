# The Ceiling of Cooperation in Extrinsic Update Rules

This repository contains the source, data-generation code, analysis code, and
manuscript for our work on heterogeneous evolutionary games. We categorise
population dynamics as _extrinsic_ (players look outward at others) or
_intrinsic_ (players look inward at their own payoffs), show that purely
extrinsic dynamics have a hard ceiling on the probability of cooperation, and
demonstrate that intrinsic dynamics can exceed it, using the public goods game
with heterogeneous contributions.

## Repository layout

The repository mirrors the structure of a reproducible data-analysis project.

- `src/public_goods_games/` is the installable package (contribution rules and
  helpers). The exact steady states and simulations are computed with the
  `ludics` library.
- `data/<experiment>/main.py` generates the raw data for an experiment and
  writes it alongside the script. `data/sweep/` holds the exhaustive
  small-population parameter sweep (archived on Zenodo) and `data/large_n/`
  holds the large-population simulations.
- `analysis/<experiment>/main.py` reads from `data/` and writes a finished
  figure into `tex/<experiment>.pdf`.
- `tex/` is flat: it contains `main.tex`, `bibliography.bib`, the standalone
  TikZ diagram sources (`*_panels.tex`), and every generated figure as a
  `.pdf`.
- `tests/` contains the test suite.

## Setup

We manage the environment with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Reproducing the data

The large-population simulations are expensive, each run is executed at most once and its
result is appended to a CSV. To (re)generate them:

```bash
uv run python data/large_n/main.py
```

Re-running the script only fills in combinations that have not been computed; to
add precision, widen the seed ranges in the script and run it again. The
small-population sweep in `data/sweep/` is large and is distributed via Zenodo
rather than regenerated routinely; it can also be reproduced from the generation
script and the job list in `data/sweep/jobs.txt`.

The small-population sweep is tracked by content hash with DVC through the
committed `*.dvc` files. The data → analysis → figure pipeline is described in
`dvc.yaml`, and `uv run dvc repro` records the resulting output hashes in
`dvc.lock`. After downloading the archived data into place, a reader can confirm
it is the exact version used here with

```bash
uv run dvc status
```

which recomputes the hashes and reports `Data and pipelines are up to date` when
everything matches.

## Building the figures

Each analysis script reads the data and writes its figure straight into `tex/`:

```bash
uv run python analysis/ceiling_extrinsic/main.py
```

The standalone diagrams are compiled directly, for example:

```bash
cd tex && pdflatex state_space_panels.tex
```

## The pipeline (DVC)

The dependencies between data, analysis, and figures are recorded in
`dvc.yaml`. To rebuild everything that is out of date:

```bash
uv run dvc repro
```

## Building the manuscript

```bash
cd tex && latexmk -pdf main.tex
```

## Tests

```bash
uv run pytest
```
