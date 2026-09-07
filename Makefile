# ProteoEM — reproducible analysis pipeline.
#
# One command:        make reproduce   (benchmarks -> analysis figures)
# List every target:  make help
#
# Every recipe mirrors the commands in TUTORIAL.md. Defaults reproduce the
# paper's numbers exactly. The manuscript PDF is built in a separate repository,
# not here; this repo reproduces the analysis and its data plots.
#
# Interpreter override (e.g. a virtualenv):
#     make PYTHON=.venv/bin/python reproduce
PYTHON ?= python
OUT    ?= outputs

.DEFAULT_GOAL := help
.PHONY: help install test smoke \
        benchmarks bench-scale bench-empirical bench-correctness bench-yield model-violation \
        figures figure-violation-cost reproduce clean

help:  ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

install:  ## Editable install with figure + test extras
	$(PYTHON) -m pip install -e ".[figures,test]"

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

smoke:  ## Fast sanity check: one small tau benchmark (seconds)
	$(PYTHON) -m proteoem.cli benchmark-tau --output $(OUT)/tau-demo --molecules 5000 --seed 7

# ---------------------------------------------------------------------------
# Benchmarks — each runner writes a self-describing archive (frozen config,
# environment + source hashes, run-status record, SHA-256 manifest).
# ---------------------------------------------------------------------------
benchmarks: bench-scale bench-empirical bench-correctness bench-yield  ## Run all reproducible benchmark harnesses

bench-scale:  ## Trace-class scale audit -> outputs/trace-class-scale
	$(PYTHON) scripts/audit_trace_class_scale.py \
	  --output $(OUT)/trace-class-scale/summary.tsv \
	  --molecules 5000 200000 1000000 --mixture-seed 7 --trace-seed 8

bench-empirical:  ## Multi-seed empirical benchmark -> outputs/empirical-benchmark
	$(PYTHON) scripts/run_empirical_benchmark.py run \
	  --output $(OUT)/empirical-benchmark \
	  --molecules 1000 5000 20000 --seeds 7 17 27 \
	  --active 32 --missing-rate 0.02 --max-iter 300 --tol 1e-7 \
	  --block-size 4096 --validation-max-molecules 5000

bench-correctness:  ## Independent correctness checks -> outputs/independent-correctness-checks
	$(PYTHON) scripts/run_independent_correctness_checks.py \
	  --output $(OUT)/independent-correctness-checks

bench-yield:  ## Observation-yield benchmark -> outputs/observation-yield-benchmark
	$(PYTHON) scripts/run_observation_yield_benchmark.py \
	  --output $(OUT)/observation-yield-benchmark \
	  --molecules 50000 --seeds 7 17 27 37 47 --max-iter 500 --tol 1e-9

model-violation:  ## Systematic 18-scenario grid, debug tier -> /tmp
	$(PYTHON) scripts/run_model_violation_benchmark.py \
	  --manifest configs/systematic_benchmark_manifest_v4.json \
	  --tier debug --output /tmp/proteoem-model-violation-v4-debug

# ---------------------------------------------------------------------------
# Analysis figures.  These are the paper's data plots.  render_abundance_parity
# and render_evidence_accumulation regenerate deterministically from the seed;
# render_observation_yield reads outputs/observation-yield-benchmark/ (run
# `bench-yield` first).  render_violation_cost (Fig 7) needs a model-violation
# archive and is a separate target below.  Schematic/illustration figures are
# built in the manuscript repository, not here.
# ---------------------------------------------------------------------------
figures:  ## Render the seed/benchmark-reproducible analysis figures
	$(PYTHON) scripts/figures/render_evidence_accumulation.py
	$(PYTHON) scripts/figures/render_abundance_parity.py
	$(PYTHON) scripts/render_observation_yield_figure.py

# Fig 7 reads a model-violation summary.tsv. Generate one (e.g. `make
# model-violation`, or a predeclared grid run) and pass it via MV_SUMMARY:
#     make figure-violation-cost MV_SUMMARY=/tmp/proteoem-model-violation-v4-debug/summary.tsv
MV_SUMMARY ?= outputs/model-violation-slim/summary.tsv
figure-violation-cost:  ## Fig 7 (needs a model-violation archive; set MV_SUMMARY)
	$(PYTHON) scripts/figures/render_violation_cost.py --summary $(MV_SUMMARY)

reproduce: benchmarks figures  ## Full pipeline: benchmarks -> analysis figures

clean:  ## Remove generated figures (keeps benchmark archives)
	rm -f figures/*.png figures/*.pdf figures/*.svg
