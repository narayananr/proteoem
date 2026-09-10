# Using ProteoEM on your own data

ProteoEM estimates how much of each candidate protein or proteoform is present in
a sample from ambiguous single-molecule affinity traces. This guide shows how to
run it on **your own data**.

(To reproduce the paper's benchmarks and plots instead, see `make reproduce` and
the README's "Reproducing the analysis" — that is a separate concern from using
the tool.)

All inference is a single call to `fit_em`. The work is preparing its three
inputs and reading its outputs.

## Runnable examples

Scripts under `examples/` walk this whole guide, so you can run it rather than
only read it:

- **`examples/try_proteoem.py`** — sections 1–7 on small hand-built arrays, one
  `print` per step. Best for seeing the API mechanics.
- **`examples/try_from_file.py`** — the same idea on a realistic tau dataset (768
  candidate proteoforms, 12 probes over 36 cycles). It writes the dataset out as
  human-readable TSVs, then reads them back and fits — the file-based workflow of
  section 8, and the closest thing to running on your own export.
- **`examples/multi_sample.py`** — quantifies a small cohort and combines the
  per-sample fits into one long/tidy `cohort_abundances.tsv` (one row per sample ×
  proteoform, in cpm); see "Quantifying multiple samples" below.

```bash
python examples/try_proteoem.py
python examples/try_from_file.py     # first run writes examples/tau_example/*.tsv, then fits
python examples/multi_sample.py      # quantifies a cohort into a long cpm table
```

To **learn the method** by simulating a sample from a known composition and
scoring the estimate against that truth, see the companion walkthrough
[`SIMULATION_TUTORIAL.md`](SIMULATION_TUTORIAL.md) and its script
`examples/simulate_and_quantify.py`. That is about understanding ProteoEM; this
guide is about running it on your own data.

## 1. What ProteoEM needs

| Input | Shape | Meaning |
|---|---|---|
| `observations` (Y) | N × C, int8 | one row per accepted molecule, one column per physical cycle; each entry is `1` (positive call), `0` (negative call), or `-1` (NA / no usable call) |
| `cycle_to_probe` | length C, int | `cycle_to_probe[c]` is the logical-probe index used in cycle `c` (repeated applications of a probe share an index) |
| `Q` | K × J, float | `Q[k, j]` is the probability that logical probe `j` gives a positive call on origin `k`, calibrated beforehand and held fixed |

`K` is the number of candidate origins, `J` the number of logical probes, `C` the
number of physical cycles (`C ≥ J` when probes repeat). Instead of `Q` you can
pass a binary feature matrix and per-probe rates (Section 3).

## 2. A minimal run

```python
import numpy as np
from proteoem import fit_em

# 3 molecules, 6 cycles = 3 logical probes applied twice.
observations = np.array([
    [1, 0, -1, 1, 0, 0],
    [0, 1,  0, 0, 1, 1],
    [1, 0,  1, 1, 0, 1],
], dtype=np.int8)
cycle_to_probe = np.array([0, 1, 2, 0, 1, 2])

# K=2 candidate origins x J=3 logical probes: calibrated positive-call rates.
Q = np.array([
    [0.90, 0.10, 0.80],
    [0.15, 0.85, 0.80],
])

fit = fit_em(observations, Q=Q, cycle_to_probe=cycle_to_probe,
             return_responsibilities=True)

print(fit.weights)            # estimated composition, sums to 1
print(fit.converged)          # check this before trusting anything
print(fit.responsibilities)   # posterior over origins, one row per trace class (T x K, T <= N)
```

`fit.weights[k]` is the estimated fraction of accepted molecules from origin `k`.

## 3. Building Q

Two routes, both calibrated on known-origin standards **before** you fit:

- **Direct.** For each origin–probe pair, `Q[k, j]` is the positive-call fraction
  of probe `j` on control molecules of known origin `k`.
- **Structured.** Give a binary feature matrix `E` (`E[k, j] = 1` if origin `k`
  carries the feature probe `j` targets) and per-probe on/off-target rates, and
  let ProteoEM build `Q`:

```python
from proteoem import build_emission_matrix
Q = build_emission_matrix(E, alpha=alpha, beta=beta)   # Q = E*alpha + (1-E)*beta
# ...or pass E straight to fit_em:
fit = fit_em(observations, E, alpha=alpha, beta=beta, cycle_to_probe=cycle_to_probe)
```

`alpha`/`beta` may be scalars or length-J vectors (per probe). The structured form
is a convenience; it does not substitute for validating that the feature model
matches your reagents.

## 4. Reading the results

`fit` (an `EMResult`) carries:

- `fit.weights` — estimated composition (K-vector, sums to 1).
- `fit.converged` — `True` if the fit met tolerance; do not interpret a
  non-converged fit.
- `fit.responsibilities` — posterior probability of each origin (columns) for each
  distinct trace class (rows), when `return_responsibilities=True`. ProteoEM groups
  identical traces, so this has one row per class (T ≤ N), not one per molecule;
  `fit.aggregated` holds the class counts and maps molecules to classes.
- `fit.observable_groups` — tuples of origin indices the panel **cannot** tell
  apart (identical emissions). Only a group's combined abundance is identifiable,
  so report the group total, `fit.weights[list(group)].sum()`, and never the split
  within a group.
- `fit.diagnostics` — `terminal_em_residual`, `monotonic`,
  `expected_count_total_error`, support density, and more.

### Per-molecule proteoform probabilities

`fit.responsibilities` is indexed by trace *class*, not by molecule. For a direct
per-molecule answer — the probability that each molecule came from each candidate —
call `posterior_responsibilities` with the fitted weights:

```python
from proteoem import posterior_responsibilities
post = posterior_responsibilities(observations, fit.weights, Q, cycle_to_probe=cycle_to_probe)
# post[i, k] = P(molecule i came from candidate k); each row sums to 1.
print(post[0])                 # molecule 0's probabilities over all candidates
print(int(post[0].argmax()))   # its most likely origin
```

`post` is N×K: one row per molecule, one column per candidate, each row a proper
distribution. Two cautions when reading a single `P(molecule i from candidate k)`:

- **Unresolvable candidates.** If candidate `k` shares emissions with others (an
  observable group), its individual probability is not identifiable. Report the
  group's combined probability from `fit.observable_group_responsibilities`, never
  the split within a group.
- **Out-of-panel molecules.** A molecule whose largest `post` value is low has no
  good match in the panel, so its true origin may lie outside your candidates. Flag
  these rather than forcing an assignment (Section 10).

## 5. Repeated probes and missing calls

Repeating a probe adds a **cycle**, not a logical probe: point the extra column at
the same `cycle_to_probe` index. ProteoEM groups the repeats by their
positive/observed counts, so their order does not matter — unless a repeat is
calibrated differently, in which case give it its own probe index and `Q` column.

An NA (`-1`) is skipped under the default ignorable-missingness model: it is not a
negative call and not a third outcome. If missingness depends on the hidden call
or the origin (bright spots saturating and suppressing positives, say), that needs
an explicit model — see the supplement's informative-missingness section.

## 6. Bring-your-own likelihood

If you compute the evidence yourself — continuous intensities, correlated cycles,
a censoring model — skip `Q` and pass a precomputed log-likelihood matrix to
`fit_likelihood_em`:

```python
from proteoem import fit_likelihood_em
fit = fit_likelihood_em(log_likelihoods, return_responsibilities=True)
```

`log_likelihoods` is N×K (log-likelihood of each molecule under each origin). Only
the across-origin comparison within a row matters, so a row-wide additive constant
is harmless; a per-origin offset is not. Classifier scores or already-normalized
posteriors are **not** likelihoods and must not be passed here.

## 7. Accepted-sample vs source composition

`fit.weights` is composition among the molecules you **fed the fitter** — the
accepted traces. If a molecule is harder to recover, or more likely to fail your
inclusion rule, it is underrepresented there. To recover the composition of the
original sample, apply an **effective observation yield** `e_k = r_k · v_k`
(physical recovery × the probability a molecule of origin `k` passes the gate):

```python
from proteoem import effective_observation_yield, analyzed_to_source_composition
e = effective_observation_yield(recovery, visibility)     # e_k = r_k * v_k
theta = analyzed_to_source_composition(fit.weights, e)    # source: theta_k proportional to pi_k / e_k
```

When a probe-based rule decides which traces are kept (e.g. "keep only traces with
a positive call"), the per-trace likelihood must **also** be conditioned on the
gate: `positive_gate_visibility(Q, ...)` gives `v_k`, and
`condition_on_deterministic_gate` renormalizes the likelihood over surviving
traces. Apply the two corrections once each — one in the likelihood, one on the
final weights — never the same one twice. Yields must be calibrated externally;
they cannot be learned from the accepted traces alone.

## 8. Applying it to real iterative-affinity data

A real iterative single-molecule affinity assay outputs exactly ProteoEM's
inputs, so the work is a thin adapter, not new modeling:

| Assay output | ProteoEM input |
|---|---|
| per-molecule binary bind/no-bind trace across cycles | `observations` (Y) |
| the cycle → antibody/probe schedule | `cycle_to_probe` |
| antibody on/off-target rates from control-proteoform calibration | `Q` |
| the candidate proteoform panel (which antibody targets which feature) | the K origins / feature matrix `E` |
| the inclusion rule (e.g. require a positive N-terminal AND a positive C-terminal call) | the retention gate — condition the likelihood, calibrate visibility (Section 7) |

Write a small function from your export format to `(observations, cycle_to_probe,
Q)` and call `fit_em`. Validate first on control mixtures of **known** composition,
where you have ground truth; treat biological-sample estimates as model output,
not truth.

## 9. Quantifying multiple samples

A single experiment already holds several samples — an IMaP flow cell runs 4–12
sample lanes — and, as in RNA-seq, **each sample is quantified on its own**: an
independent fit, one output file per sample. There is no joint multi-sample model.

Every fit writes the same **general, truth-free** file, `proteoform_abundances.tsv`,
in the paper's cpm unit, carrying **flow-cell and lane** provenance so a sample
traces back to its physical position:

```
flowcell  lane  sample   sample_group  candidate_id                              estimated_cpm
fc1       1     brain_A  control       1N3R|pT181+pS202_pT205+pS214+pT217+pS396  175599
fc1       1     brain_A  control       0N4R|pT181+pS202_pT205                    109925
...   (every candidate is written, like an RNA-seq quantifier; threshold downstream)
```

The CLI writes it per run, labelled with `--sample`, `--lane`, and `--flowcell`:

```bash
proteoem benchmark-tau --output out/brain_A --sample brain_A --lane 1 --sample-group control --seed 7
proteoem benchmark-tau --output out/brain_B --sample brain_B --lane 2 --sample-group disease --seed 8
```

To build a cohort, concatenate the per-sample files — like assembling an RNA-seq
count matrix — giving one row per (sample, proteoform), ready to pivot to a
proteoform × sample matrix for z-scoring, clustering, or differential tests.
`examples/multi_sample.py` does this and writes `cohort_abundances.tsv`.

**Truth stays out of it.** `proteoform_abundances.tsv` is the estimate only, so it
is identical for real and simulated data. Ground truth exists only in simulation
and lives in a separate file (`true_abundances.tsv`, or the benchmark's
`benchmark_comparison.tsv`) — never in the general output.

## 10. What to check

- `fit.converged` is `True`.
- Report `fit.observable_groups` totals; never interpret within-group splits.
- Flag trace classes whose maximum `fit.responsibilities` is low — those are the
  molecules whose true origin may be **outside** your candidate panel.
- Check calibration (expected calibration error, Brier score) against control
  mixtures before trusting a biological sample.

ProteoEM is currently evaluated in simulation only; it has not been validated on
real molecule-level data. Use it as a transparent, auditable estimator, and
establish its behavior on your own controls.

## Command line

For a quick end-to-end check without writing code, the CLI runs one synthetic
tau-like dataset through the whole pipeline:

```bash
proteoem benchmark-tau --output outputs/tau-demo --molecules 5000 --seed 7
# vary the simulation: --concentration, --repeats, --active, --missing-rate, --molecules
```

See `proteoem benchmark-tau --help` for all flags.
