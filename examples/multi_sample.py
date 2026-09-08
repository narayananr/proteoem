#!/usr/bin/env python3
"""
multi_sample.py — quantify a COHORT of samples and collect the result into one
long/tidy table, the way the IMaP paper reports many samples.

    python examples/multi_sample.py

ProteoEM fits one sample at a time -- the assay estimates per-sample antibody
behaviour, so each sample is an independent fit and there is no joint model. To
quantify a cohort you loop over samples, fit each, and append its proteoform
abundances (in cpm) to a long table keyed by sample. The schema

    sample    sample_group    candidate    estimated_cpm

mirrors the paper's released format (sample, sample_group, proteoform,
abundance_cpm): one row per (sample, proteoform), which is what scales to a
cohort and is ready for cross-sample z-scoring, clustering, or differential tests.
"""
import csv
from pathlib import Path
import numpy as np
from proteoem import run_tau_like_benchmark

HERE = Path(__file__).resolve().parent
OUT = HERE / "tau_example"
OUT.mkdir(parents=True, exist_ok=True)

# A small cohort. Different seeds give different true compositions, standing in
# for different biological samples (each with its own present proteoforms).
SAMPLES = [
    ("healthy_1", "control", 7),
    ("healthy_2", "control", 17),
    ("disease_1", "ADRD", 27),
]
N_MOLECULES = 3000

rows = []
for name, group, seed in SAMPLES:
    # one independent fit per sample (as the paper does)
    r = run_tau_like_benchmark(n_molecules=N_MOLECULES, n_active=32,
                               missing_rate=0.02, seed=seed)
    est = np.asarray(r.weighted_fit.weights)         # fraction, sums to 1
    cand = r.panel.candidate_ids
    n_detected = 0
    for k in np.argsort(est)[::-1]:
        if est[k] * N_MOLECULES >= 20:               # detected in THIS sample
            rows.append((name, group, cand[k], int(round(est[k] * 1e6))))  # cpm
            n_detected += 1
    print(f"  fit {name:11s} ({group:7s}, seed {seed:>2d}): {n_detected} proteoforms detected")

out_path = OUT / "proteoform_abundances.tsv"
with out_path.open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["sample", "sample_group", "candidate", "estimated_cpm"])
    w.writerows(rows)

print(f"\nwrote {out_path.name}: {len(rows)} sample-proteoform rows across {len(SAMPLES)} samples")
print("\npeek:")
print(f"  {'sample':11s}  {'group':7s}  {'candidate':44s} {'cpm':>7s}")
for row in rows[:6]:
    print(f"  {row[0]:11s}  {row[1]:7s}  {row[2]:44s} {row[3]:7d}")

print("\nThis long/tidy shape scales to a cohort: one row per (sample, proteoform),"
      "\nmatching the paper's release (sample / sample_group / proteoform / abundance_cpm)."
      "\nPivot it to a proteoform x sample matrix for heatmaps, clustering, or"
      "\ndifferential-abundance tests across samples.")
