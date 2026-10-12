# Lorentz retrieval: a reproducible evidence deck

This repository is a compact, self-contained release of the final method and
the evidence used in the accompanying Beamer slides.  It intentionally does
not claim that hyperbolic retrieval wins universally.

The main empirical finding is narrower:

* a row-normalized direct-Lorentz head with full-training LoRA is competitive
  on the three-task development suite and has a local 128-D advantage;
* on the frozen MMEB-v3 evaluation, the broad 97-task bounded screen does not
  improve overall;
* NanoSciFact is the only pre-registered full-corpus task with a reproducible
  positive effect at both 128 and 256 dimensions.

## Reproduce everything

Only Python 3 and a LaTeX distribution with `beamer`, `booktabs`, `tikz`, and
`pgfplots` are required.

```bash
make test
make reproduce
```

The second command regenerates every table/plot fragment from the frozen CSV
files and builds `slides/main.pdf`.  To verify without LaTeX:

```bash
python scripts/generate_assets.py --check
```

A no-dependency geometry demo is available as:

```bash
PYTHONPATH=src python scripts/demo.py
sha256sum -c SHA256SUMS
```

## Repository layout

* `src/lorentz_retrieval/geometry.py`: dependency-free Lorentz lift, distance,
  and hybrid score reference implementation.
* `src/lorentz_retrieval/head.py`: optional PyTorch implementation of the
  trained output head.
* `data/`: frozen per-seed summaries and provenance hashes.
* `config/h10_recipe.json`: the exact final H10 and matched Euclidean recipes.
* `scripts/generate_assets.py`: deterministic table/plot generation.
* `slides/main.tex`: the short scientific deck.

The large Gemma-4 checkpoint and MMEB-v3 corpora are not redistributed.  The
configuration records their identifiers and the BSC execution recipe, while
all reported numbers can be independently regenerated from the included
frozen summaries.

The training recipe is fully specified, but reproducing model training still
requires separately licensed Gemma-4 weights, MMEB-train/MMEB-v3 data, GPUs,
and the corresponding preprocessing stack. “Self-contained” here means that
the geometry implementation, evidence validation, plots, tables, and slides
require no private source tree or cluster paths.

## Scientific guardrails

Cosine, Euclidean distance, and Lorentz distance do not share an absolute
margin scale.  This release compares official/rank-based metrics, signs of
pairwise margins, and within-geometry changes.  Raw cross-geometry distance
magnitudes are deliberately not used as evidence.

## License

Code is MIT licensed.  Numerical results retain the terms of their underlying
datasets and model checkpoints.
