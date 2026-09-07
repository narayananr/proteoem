# ProteoEM: step-by-step tutorial

This walks through every analysis script in the repository one at a time: what it
does, the exact command, what it writes, and roughly how long it takes.

This repository reproduces the **analysis** — the benchmarks and their data plots.
The manuscript PDF and the schematic/illustration figures are built in a separate
manuscript repository and are not part of this code repo.

For a single command that runs the whole analysis pipeline, use `make reproduce`
(see the end of this file). This tutorial is for running and understanding each
piece on its own.

**Interpreter note.** The commands below assume your Python environment is
active. If it is not, prefix the interpreter path, for example
`.venv/bin/python` instead of `python`, or pass it to `make`:
`make PYTHON=.venv/bin/python <target>`.

All data are simulated. Nothing here is Nautilus data or a validation of any
commercial assay.

---

## 0. Set up (once)

```bash
python -m pip install -e ".[figures,test]"   # package + figure/test extras
python -m pytest -q                           # optional sanity check
```

`make install` runs the first line; `make test` runs the second.

---

## 1. Quick synthetic benchmark — the CLI

**What it does.** Simulates one tau-like dataset (768 candidate proteoforms, 12
logical probes over 36 cycles) and fits weighted EM plus the two reduced
baselines. The fastest way to see the method run end to end.

```bash
python -m proteoem.cli benchmark-tau --output outputs/tau-demo --molecules 5000 --seed 7
```

**Writes.** `outputs/tau-demo/`: `summary.json` (metrics, config, environment),
`abundances.tsv` (truth vs. each estimate), `panel.tsv`. Add `--save-traces` for
the raw molecule-by-cycle matrix and simulation truth.

**Run with other parameters.** All knobs are exposed:

```bash
# flatter composition (fewer rare states) and only two passes per probe
python -m proteoem.cli benchmark-tau --output outputs/tau-flat \
  --molecules 20000 --seed 7 --concentration 1.0 --repeats 2 --active 40
```

| Flag | Meaning | Default |
|---|---|---|
| `--molecules` | accepted traces to simulate | 5000 |
| `--active` | present proteoforms (rest are zero) | 32 |
| `--concentration` | Dirichlet concentration on active states (`<1` heavy-tailed, larger flatter) | 0.4 |
| `--repeats` | physical passes per logical probe (`cycles = 12 × repeats`) | 3 |
| `--missing-rate` | fraction of calls set to NA | 0.02 |
| `--seed` | reproducible RNG seed | 7 |
| `--max-iter`, `--tol`, `--block-size` | EM stopping and blocking | 300, 1e-7, 4096 |

**Time:** seconds. `make smoke` runs the default command.

---

## 2. Trace-class scale audit

**What it does.** Counts identical, exchangeable-repeat, and relative
trace-likelihood classes as the molecule count grows, showing how grouping
compresses the E-step and why the affinity likelihood table stays dense.

```bash
python scripts/audit_trace_class_scale.py \
  --output outputs/trace-class-scale/summary.tsv \
  --molecules 5000 200000 1000000 --mixture-seed 7 --trace-seed 8
```

**Writes.** `outputs/trace-class-scale/summary.tsv`. **Time:** ~1–2 min (the
1,000,000 count dominates). `make bench-scale`.

---

## 3. Multi-seed empirical benchmark

**What it does.** Runs the correctly specified design across molecule counts and
seeds, each in a fresh process, recording per-run class sizes, timings, peak
memory, convergence, and abundance metrics.

```bash
python scripts/run_empirical_benchmark.py run \
  --output outputs/empirical-benchmark \
  --molecules 1000 5000 20000 --seeds 7 17 27 \
  --active 32 --missing-rate 0.02 --max-iter 300 --tol 1e-7 \
  --block-size 4096 --validation-max-molecules 5000
```

**Writes.** `outputs/empirical-benchmark/`: per-run and summary tables, likelihood
histories, exact-compression checks, environment metadata, source hashes.
**Time:** ~2–5 min. `make bench-empirical`. `--molecules` and `--seeds` take
lists, so `--molecules 1000 --seeds 7` gives a quick check.

---

## 4. Independent correctness checks

**What it does.** Compares fixed-emission EM against a separate projected-gradient
simplex optimizer and a separately implemented incidence EM, and runs a
predeclared likelihood-ratio threshold sweep.

```bash
python scripts/run_independent_correctness_checks.py \
  --output outputs/independent-correctness-checks
```

**Writes.** `outputs/independent-correctness-checks/`: optimizer and
deterministic-binary comparisons, all threshold-sweep runs, frozen config,
environment metadata, source hashes, summary. **Time:** ~1–3 min.
`make bench-correctness`.

---

## 5. Observation-yield benchmark (and its figure)

**What it does.** A two-origin study separating accepted-trace composition (`π`)
from source composition (`θ`) under three keep-rules, plus a confidence-filtered
hard decode as a biased comparator.

```bash
python scripts/run_observation_yield_benchmark.py \
  --output outputs/observation-yield-benchmark \
  --molecules 50000 --seeds 7 17 27 37 47 --max-iter 500 --tol 1e-9

python scripts/render_observation_yield_figure.py     # reads the archive above
```

**Writes.** `outputs/observation-yield-benchmark/` and
`figures/figure5_observation_yield_benchmark.svg`. **Time:** ~1–3 min for the
benchmark, seconds for the figure. `make bench-yield`, then `make figures`.

> The figure reads `outputs/observation-yield-benchmark/`, so run the benchmark
> first (or point `--input` at an existing archive).

---

## 6. Systematic model-violation grid

**What it does.** Runs the frozen 18-scenario grid (missingness, Q error, repeat
dependence, identifiability, candidate omission, OOD artifacts). The runner
consumes the frozen manifest directly and never silently substitutes a smaller
run.

```bash
python scripts/run_model_violation_benchmark.py \
  --manifest configs/systematic_benchmark_manifest_v4.json \
  --tier debug --output /tmp/proteoem-model-violation-v4-debug
```

**Writes.** A self-describing archive under the chosen `--output`.
**Time:** several minutes at the `debug` tier.

- `--tier debug` is the runnable smoke of the grid; `--tier primary` is the full
  authoritative run and needs a clean Git snapshot.
- Narrow the run with `--scenario-id`, `--master-seed`, or `--method-id`.
- `make model-violation` runs the debug-tier command above.

---

## 7. Analysis figures

**What it does.** Renders the paper's data plots.

```bash
python scripts/figures/render_evidence_accumulation.py  # posterior evidence over probing
python scripts/figures/render_abundance_parity.py       # estimated vs true abundance
python scripts/render_observation_yield_figure.py       # observation-yield results
```

**Writes.** `figures/<name>.png` and `.pdf` (each matplotlib script takes
`--output` as a path stem and `--seed`). **Time:** seconds each. `make figures`.

- `render_evidence_accumulation` and `render_abundance_parity` regenerate their
  data deterministically from the seed — no benchmark output needed.
- `render_observation_yield_figure` reads the Step 5 archive.
- **Fig 7 (model-violation cost)** reads a model-violation `summary.tsv`, so it is
  a separate step. Generate an archive (Step 6) and point the script at it:

  ```bash
  make model-violation
  python scripts/figures/render_violation_cost.py \
    --summary /tmp/proteoem-model-violation-v4-debug/summary.tsv
  # or: make figure-violation-cost MV_SUMMARY=/tmp/proteoem-model-violation-v4-debug/summary.tsv
  ```

- The schematic/illustration figures (measurement-to-inference, trace-to-likelihood,
  information-borrowing, the generative-model pipeline, panel identifiability) are
  built in the manuscript repository, not here.

---

## Run order and dependencies

```
benchmarks (Steps 2–6) ──▶ analysis figures (Step 7)
                            ▲
   observation-yield figure needs the Step 5 archive; Fig 7 needs a Step 6 archive
```

Steps 1–6 can run in any order. `render_evidence_accumulation` and
`render_abundance_parity` need nothing but the seed.

## One command

```bash
make reproduce                            # benchmarks -> analysis figures
make help                                 # list every target
make PYTHON=.venv/bin/python reproduce    # use a specific interpreter
```

`make reproduce` runs the reproducible benchmarks, then the seed/benchmark-driven
analysis figures. Fig 7 is the separate `make figure-violation-cost` target
because it needs a model-violation archive.
