# ProteoEM

[![CI](https://github.com/narayananr/proteoem/actions/workflows/ci.yml/badge.svg)](https://github.com/narayananr/proteoem/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22721009.svg)](https://doi.org/10.5281/zenodo.22721009)

ProteoEM quantifies proteins and proteoforms from single-molecule affinity traces.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/method-dark@2x.png">
  <img alt="Four-panel schematic of the ProteoEM model. One molecule is read by six
probe cycles applied in order, giving a trace of positive, negative and missing calls.
Three such traces are shown, with a missing call skipped rather than counted as a
negative. An emission matrix, measured in advance from known proteoforms and held fixed
during the fit, gives each proteoform's probability of a positive call for each probe.
The two proteoforms behave alike, so one trace cannot tell them apart. Each trace
therefore counts as one molecule whose count is shared between the proteoforms in
proportion to its posterior responsibility, rather than given to either one outright.
Adding up a column gives the number of molecules for that proteoform, and dividing by
the three molecules gives each proteoform's share."
       src="docs/figures/method-light@2x.png">
</picture>

It treats each trace as probabilistic evidence over candidate origins, keeps
that evidence graded instead of forcing a single identity, and combines the
traces by expectation-maximization (EM) using a probe emission model calibrated
beforehand from known origins.

- Pure Python, one runtime dependency (NumPy).
- A likelihood interface plus an EM kernel that handles both calibrated
  affinity likelihoods and the binary-compatibility special case.
- A command-line synthetic benchmark for reproducible experiments.

## The problem it solves

Iterative single-molecule affinity mapping reads a protein one molecule at a
time: a series of affinity probes is applied to a molecule held at a fixed
location, and each probe reports a yes/no binding call. The ordered calls form
an **affinity trace**. Because probes bind imperfectly and different proteins
can share features, a single trace is usually consistent with *several*
candidate proteins or proteoforms. Estimating how much of each is present
therefore means assigning ambiguous traces across candidates and counting — a
mixture-inference problem, not a lookup.

This is the same shape as **RNA-seq transcript quantification**, where a
sequencing read can come from several transcripts and EM estimates transcript
abundances by splitting each read's count among its possible sources. ProteoEM
applies that idea to affinity traces: it scores each trace against every
candidate using calibrated, externally fixed probe-response rates, then splits
each molecule's single count in proportion to how well each candidate explains
it. Because each intact molecule is read on its own, the features seen together
define its proteoform, so ProteoEM can resolve proteoforms, not only proteins.

The correspondence with RNA-seq, and where it breaks down (affinity evidence is
*dense* rather than sparse, and requires an externally calibrated emission
matrix), is what motivates the model.

## Scope

ProteoEM is evaluated in simulation. Its tau-like data are synthetic, drawn
from publicly described design features. They are not Nautilus data and do not
reproduce any proprietary software. Benchmark error, memory, and runtime values
characterize the estimator, not the analytical performance of any commercial
platform.

## Requirements

- Python 3.10 or later
- NumPy 1.24 or later (installed automatically)

## Install

Not yet on PyPI. Install from a source checkout:

```bash
git clone https://github.com/narayananr/proteoem.git
cd proteoem
python -m pip install .
```

For an editable install with the test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The wheel provides the importable `proteoem` package and the `proteoem`
command-line entry point.

## Quick start

Estimate a mixture from a small call matrix. Calls are `1` (positive), `0`
(negative), and `-1` (unavailable/NA):

```python
import numpy as np
from proteoem import fit_em

# Three molecules, six physical cycles applying three logical probes twice.
observations = np.array([
    [1, 0, -1, 1, 0, 0],
    [0, 1,  0, 0, 1, 1],
    [1, 0,  1, 1, 0, 1],
], dtype=np.int8)
cycle_to_probe = np.array([0, 1, 2, 0, 1, 2])

# Calibrated origin-by-logical-probe positive-call probabilities (freeze before fitting).
Q = np.array([
    [0.90, 0.10, 0.80],
    [0.15, 0.85, 0.80],
])

fit = fit_em(observations, Q=Q, cycle_to_probe=cycle_to_probe)

print(fit.weights)     # -> [0.6665 0.3335]
print(fit.converged)   # -> True
```

Check `fit.converged` before interpreting estimates. Useful diagnostics live in
`fit.diagnostics` (for example `terminal_em_residual`, `monotonic`,
`expected_count_total_error`).

Run the bundled synthetic benchmark from the command line:

```bash
proteoem benchmark-tau --output outputs/tau-demo --molecules 5000 --seed 7
```

This writes `proteoform_abundances.tsv` (your abundances, one row per proteoform
in cpm) plus `benchmark_comparison.tsv` (the simulation benchmark: truth and the
baseline methods), `panel.tsv`, and `summary.json` under the output directory. Add `--save-traces` to also retain the generated molecule-by-cycle
matrix and simulation truth. See `proteoem benchmark-tau --help` for all flags.

For a fuller guide to using ProteoEM on your own data — preparing the inputs,
calibrating `Q`, the accepted-versus-source yield correction, and reading the
results — see [`TUTORIAL.md`](TUTORIAL.md).

## How it works

Let `Y` be a molecule-by-physical-cycle call matrix and let `cycle_to_probe[c]`
give the zero-based logical-probe index used in physical cycle `c`. An NA is
marginalized under the default ignorable-missingness model; it is not recoded as
a negative call and is not treated as a third outcome. If missingness depends on
the unseen call or on molecular origin, a separate censoring or categorical
model is required.

For calibrated binary emissions, `Q[k, j]` is the probability of a positive call
from logical probe `j` when the origin is `k`. Estimate `Q` before analyzing the
unknown mixture and freeze it during abundance EM; the unknown mixture is never
used to learn `Q`. Valid construction routes are (1) direct origin-by-probe
positive-call estimates from adequate known-origin standards or (2) an
independently validated structured model such as `Q = E * alpha + (1 - E) * beta`.
ProteoEM forms

```text
log L[i, k] = log P(observed trace i | origin k, fixed Q)
```

and estimates the mixture by EM. A binary compatibility profile (`0` compatible,
`-inf` incompatible) is the RNA-seq-style special case, and the same EM kernel
handles both forms of evidence.

**Composition among accepted traces is not source composition.** The fit
estimates composition among the traces supplied to it. Recovering source
composition requires a separately calibrated effective observation yield
`e[k] = physical_recovery[k] * panel_visibility[k]`, with
`source[k] ∝ accepted[k] / e[k]`. When a probe-based gate decides which traces
are retained, the gate must be modeled explicitly rather than corrected by a
protein-length or probe-count divisor. The `proteoem.observation_yield` module
implements this correction, with the gate-conditioning and inverse-yield helpers.

**Observable groups.** Origins with exactly identical fixed emissions cannot be
separated by any probe in the panel. `fit_em` finds these groups and reports
their combined estimates (`fit.observable_groups`,
`fit.observable_group_weights`); interpret group totals, not candidate-level
splits within a group.

## Command-line interface

```text
proteoem benchmark-tau   run the 768-candidate synthetic tau-like benchmark
```

Comparative abundance metrics are convergence-gated: they are reported only when
both the weighted-affinity and binary-incidence EM fits converge, otherwise
`metrics_valid` is false and `metrics` is null in the output.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Reproducing the analysis

The benchmarks and their data plots reproduce from `scripts/`. For a single
command, run `make reproduce` (benchmarks, then the analysis figures); `make
help` lists every target. The manuscript PDF and the schematic figures are built
in a separate manuscript repository, not here.

To use ProteoEM as a tool on your own data — rather than reproduce the paper —
see [`TUTORIAL.md`](TUTORIAL.md). To learn the method by simulating a sample from
a known composition and scoring the estimate against it, see
[`SIMULATION_TUTORIAL.md`](SIMULATION_TUTORIAL.md) (run it with
`examples/simulate_and_quantify.py --plots`).

Frozen benchmark manifests used by the model-violation grid live under
`configs/`.

The method schematic in this file is generated, not drawn:

```bash
python -m pip install -e ".[figures]"
python docs/figures/make_method_figure.py
```

Every number in it is computed by calling `fit_em`, and the script asserts
convergence, that no probe was dropped as uninformative, that each molecule's
responsibilities sum to one, and that they accumulate to the reported expected
counts. It fails rather than emitting a figure that disagrees with the model.
Both themes and the vector versions are written to `docs/figures/`.

`make_method_figure.py` also writes a three-panel `measurement-*` variant, panels
A to C only. Panel D shows a fitted split, so writing that has not walked through
the fit yet should use that version rather than leave the numbers unexplained.

`docs/figures/make_platform_figure.py` renders a companion schematic of the
measurement itself, for talks and supplementary material. Its output is not
committed, so run the script to produce it. Both scripts share one palette and
one set of drawing helpers, in `docs/figures/figstyle.py`.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use ProteoEM, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).
