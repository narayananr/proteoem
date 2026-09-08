#!/usr/bin/env python3
"""
try_from_file.py — the realistic version of the tutorial: load a dataset from
FILES and fit it, instead of hand-typed toy arrays.

    python examples/try_from_file.py          # from a repo checkout, with proteoem installed

On first run it writes a synthetic tau-panel dataset (768 candidate proteoforms,
12 probes over 36 cycles, calibrated per-probe rates) to human-readable TSVs under
examples/tau_example/ -- your stand-in for a real assay export. Then it reads
those files back and runs one fit_em(), the way you would with your own data.

The files it uses (this is the shape a real export would have):
  traces.tsv          N molecules x 36 cycles, values 1 / 0 / -1 (positive/neg/NA)
  probe_schedule.tsv  which logical probe each of the 36 cycles used
  emission_Q.tsv      768 candidates x 12 probes, the calibrated positive-call rates
  true_abundance.tsv  the ground truth in cpm (a real dataset would NOT have this)

It writes the fit back out as proteoform_abundances.tsv in counts per million
(cpm = fraction * 1e6, the IMaP paper's unit) -- the general, truth-free output,
one file per sample, like an RNA-seq quantifier.
"""
import csv
from pathlib import Path
import numpy as np

from proteoem import (
    fit_em,
    make_tau_like_panel, default_tau_probe_rates, default_tau_logical_probe_rates,
    sparse_tau_weights, build_emission_matrix,
)
from proteoem import simulate_traces

HERE = Path(__file__).resolve().parent
DATA = HERE / "tau_example"
N_MOLECULES = 3000


# ---------------------------------------------------------------------------
# One-time: write the dataset out as files (your stand-in for a real export).
# ---------------------------------------------------------------------------
def generate_example_files(datadir: Path) -> None:
    datadir.mkdir(parents=True, exist_ok=True)
    panel = make_tau_like_panel(repeats=3)
    alpha, beta = default_tau_probe_rates(panel)                    # per cycle (36)
    logical_alpha, logical_beta = default_tau_logical_probe_rates(panel)  # per probe (12)
    truth = sparse_tau_weights(panel, n_active=32, seed=7)          # 768, 32 nonzero
    sim = simulate_traces(panel.profiles, N_MOLECULES, weights=truth,
                          alpha=alpha, beta=beta, missing_rate=0.02, seed=8)

    obs = np.asarray(sim.observations)                             # (N, 36) in {-1,0,1}
    c2p = np.asarray(panel.cycle_to_probe)                         # (36,) probe index per cycle
    Q = build_emission_matrix(panel.logical_profiles,
                              alpha=logical_alpha, beta=logical_beta)   # (768, 12)
    probes = list(panel.logical_probe_ids)                        # 12 names
    cand = list(panel.candidate_ids)                              # 768 names

    with (datadir / "traces.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["molecule"] + [f"c{j+1:02d}" for j in range(obs.shape[1])])
        for i, row in enumerate(obs):
            w.writerow([f"mol_{i:05d}"] + [int(v) for v in row])

    with (datadir / "probe_schedule.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cycle", "logical_probe"])
        for c, p in enumerate(c2p):
            w.writerow([f"c{c+1:02d}", probes[int(p)]])

    with (datadir / "emission_Q.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["candidate"] + probes)
        for name, qrow in zip(cand, Q):
            w.writerow([name] + [f"{v:.4f}" for v in qrow])

    with (datadir / "true_abundance.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["candidate", "true_cpm"])   # counts per million, the unit the paper reports
        for name, t in zip(cand, truth):
            if t > 0:
                w.writerow([name, int(round(t * 1e6))])
    print(f"[generated {N_MOLECULES}-molecule tau dataset in {datadir}/]")


# ---------------------------------------------------------------------------
# The part a real user does: read the files, assemble the three inputs, fit.
# ---------------------------------------------------------------------------
def load_traces(path: Path):
    with path.open() as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        ids, rows = [], []
        for row in r:
            ids.append(row[0]); rows.append([int(x) for x in row[1:]])
    return ids, np.array(rows, dtype=np.int8)


def load_Q(path: Path):
    with path.open() as f:
        r = csv.reader(f, delimiter="\t"); header = next(r)
        probes = header[1:]; names, rows = [], []
        for row in r:
            names.append(row[0]); rows.append([float(x) for x in row[1:]])
    return names, probes, np.array(rows, dtype=float)


def load_schedule(path: Path, probes):
    idx = {p: j for j, p in enumerate(probes)}
    with path.open() as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        return np.array([idx[row[1]] for row in r], dtype=int)


def load_truth(path: Path):
    # the file is in cpm (counts per million); convert back to a fraction for the math
    out = {}
    with path.open() as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            out[row[0]] = float(row[1]) / 1e6
    return out


def main() -> None:
    if not (DATA / "traces.tsv").exists():
        generate_example_files(DATA)

    # --- read the files ----------------------------------------------------
    mol_ids, observations = load_traces(DATA / "traces.tsv")
    cand_ids, probes, Q = load_Q(DATA / "emission_Q.tsv")
    cycle_to_probe = load_schedule(DATA / "probe_schedule.tsv", probes)

    print("\nLoaded from files:")
    print(f"  traces        : {observations.shape[0]} molecules x {observations.shape[1]} cycles")
    print(f"  emission_Q    : {Q.shape[0]} candidates x {Q.shape[1]} probes")
    print(f"  probe schedule: 36 cycles -> {len(set(cycle_to_probe))} logical probes")
    print("  first trace   :", observations[0].tolist())

    # --- fit ---------------------------------------------------------------
    fit = fit_em(observations, Q=Q, cycle_to_probe=cycle_to_probe)
    est = fit.weights
    print(f"\nfit: converged={fit.converged} in {fit.n_iter} iters, "
          f"{sum(1 for w in est if w * observations.shape[0] >= 20)} proteoforms detected "
          f"(expected >=20 molecules).")

    # --- save the estimate in the paper's unit -----------------------------
    # fit.weights is a fraction (sums to 1).  counts-per-million = fraction * 1e6,
    # exactly how the IMaP paper reports abundances.  This is the general, truth-free
    # output: one file per sample, like an RNA-seq quantifier.
    est_cpm = est * 1e6
    out_path = DATA / "proteoform_abundances.tsv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["flowcell", "lane", "sample", "candidate", "estimated_cpm"])
        for k in np.argsort(est)[::-1]:
            if est[k] * observations.shape[0] >= 20:      # detected proteoforms only
                w.writerow(["fc1", "1", "sample_1", cand_ids[k], int(round(est_cpm[k]))])
    print(f"wrote {out_path.name}  (general, truth-free abundances in cpm; one file per sample)")

    # --- read off the answer, compared to the (known) truth ----------------
    truth = load_truth(DATA / "true_abundance.tsv")
    truth_vec = np.array([truth.get(c, 0.0) for c in cand_ids])
    tv = 0.5 * float(np.sum(np.abs(est - truth_vec)))     # TV is computed on fractions
    order = np.argsort(est)[::-1][:10]
    print(f"\nTop proteoforms recovered (total-variation error vs truth: {tv:.4f}):")
    print(f"  {'candidate':44s} {'est cpm':>10s} {'true cpm':>10s}")
    for k in order:
        print(f"  {cand_ids[k]:44s} {int(round(est_cpm[k])):10d} {int(round(truth_vec[k]*1e6)):10d}")

    print("\nThat is the whole real workflow: read (observations, cycle_to_probe, Q)"
          "\nfrom files, call fit_em, read off fit.weights, save it as cpm. Swap in your"
          "\nown export and nothing else changes. See TUTORIAL.md for the format.")


if __name__ == "__main__":
    main()
