"""Command-line entry points for reproducible synthetic benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from . import __version__
from .benchmark import run_tau_like_benchmark


def _write_benchmark(
    result,
    output: Path,
    *,
    save_traces: bool,
    configuration: dict[str, object],
    sample: str = "sample_1",
    flowcell: str = "fc1",
    lane: str = "1",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    truth = result.simulation.weights

    with (output / "abundances.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "candidate_index",
                "candidate_id",
                "truth",
                "hard_ml",
                "binary_incidence_em",
                "weighted_affinity_em",
                "truth_cpm",
                "hard_ml_cpm",
                "binary_incidence_em_cpm",
                "weighted_affinity_em_cpm",
            ]
        )
        for candidate, candidate_id in enumerate(result.panel.candidate_ids):
            t = float(truth[candidate])
            h = float(result.hard_weights[candidate])
            b = float(result.incidence_fit.weights[candidate])
            w = float(result.weighted_fit.weights[candidate])
            # counts per million = fraction * 1e6, the unit the IMaP paper reports
            writer.writerow(
                [
                    candidate,
                    candidate_id,
                    f"{t:.12g}",
                    f"{h:.12g}",
                    f"{b:.12g}",
                    f"{w:.12g}",
                    round(t * 1e6),
                    round(h * 1e6),
                    round(b * 1e6),
                    round(w * 1e6),
                ]
            )

    # General, truth-free abundance file: the estimate only, in cpm, one file per
    # sample (like an RNA-seq quantifier). This is the shape that also fits real
    # data; truth lives only in abundances.tsv, which a simulation benchmark has.
    weighted = result.weighted_fit.weights
    with (output / "proteoform_abundances.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["flowcell", "lane", "sample", "candidate", "estimated_cpm"])
        ranked = sorted(range(len(weighted)), key=lambda k: float(weighted[k]), reverse=True)
        for candidate in ranked:
            cpm = int(round(float(weighted[candidate]) * 1e6))
            if cpm >= 1:
                writer.writerow(
                    [flowcell, lane, sample, result.panel.candidate_ids[candidate], cpm]
                )

    with (output / "panel.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["candidate_index", "candidate_id", *result.panel.probe_ids])
        for candidate, candidate_id in enumerate(result.panel.candidate_ids):
            writer.writerow(
                [candidate, candidate_id, *result.panel.profiles[candidate].tolist()]
            )

    summary = {
        "data_origin": "synthetic_design_inspired",
        "warning": (
            "This benchmark is not Nautilus data and does not reproduce or validate "
            "a commercial platform or proprietary decoder."
        ),
        "n_molecules": int(result.simulation.observations.shape[0]),
        "n_candidates": result.panel.n_candidates,
        "n_cycles": result.panel.n_probes,
        "n_logical_probes": result.panel.n_logical_probes,
        "n_unique_trace_mask_patterns": result.simulation.aggregated.n_unique,
        "n_repeated_probe_sufficient_profiles": result.weighted_fit.diagnostics[
            "n_sufficient_count_classes"
        ],
        "n_trace_likelihood_classes": result.weighted_fit.diagnostics[
            "n_trace_likelihood_classes"
        ],
        "missing_fraction": result.simulation.aggregated.missing_fraction,
        "metrics_valid": result.metrics_valid,
        "metrics": result.metrics if result.metrics_valid else None,
        "configuration": configuration,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "weighted_fit": {
            "converged": result.weighted_fit.converged,
            "iterations": result.weighted_fit.n_iter,
            "log_likelihood": result.weighted_fit.log_likelihood,
            "monotonic": result.weighted_fit.diagnostics["monotonic"],
            "terminal_em_residual": result.weighted_fit.diagnostics[
                "terminal_em_residual"
            ],
            "expected_count_total_error": result.weighted_fit.diagnostics[
                "expected_count_total_error"
            ],
            "likelihood_support_density": result.weighted_fit.diagnostics[
                "likelihood_support_density"
            ],
            "observable_groups": result.weighted_fit.diagnostics[
                "n_observable_groups"
            ],
            "unresolved_groups": result.weighted_fit.diagnostics[
                "n_unresolved_groups"
            ],
        },
        "binary_incidence_fit": {
            "converged": result.incidence_fit.converged,
            "iterations": result.incidence_fit.n_iter,
            "log_likelihood": result.incidence_fit.log_likelihood,
            "monotonic": result.incidence_fit.diagnostics["monotonic"],
            "terminal_em_residual": result.incidence_fit.diagnostics[
                "terminal_em_residual"
            ],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if save_traces:
        np.savez_compressed(
            output / "synthetic_traces.npz",
            observations=result.simulation.observations,
            true_identities=result.simulation.identities,
            true_weights=result.simulation.weights,
            emission_probabilities=result.simulation.Q,
            logical_emission_probabilities=result.weighted_fit.Q,
            cycle_to_probe=result.panel.cycle_to_probe,
            candidate_profiles=result.panel.profiles,
            logical_candidate_profiles=result.panel.logical_profiles,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteoem",
        description="Inference and synthetic benchmarks for iterative affinity traces",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser(
        "benchmark-tau",
        help="run the public-design-inspired 768-candidate synthetic benchmark",
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--molecules", type=int, default=5_000)
    benchmark.add_argument("--active", type=int, default=32)
    benchmark.add_argument(
        "--concentration",
        type=float,
        default=0.4,
        help="symmetric Dirichlet concentration on active states "
        "(<1 heavy-tailed, larger flatter; default 0.4 matches the frozen benchmark)",
    )
    benchmark.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="physical passes per logical probe (cycles = 12 x repeats; default 3)",
    )
    benchmark.add_argument("--missing-rate", type=float, default=0.02)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument(
        "--sample",
        default="sample_1",
        help="sample label for the general, truth-free proteoform_abundances.tsv "
        "output (run with different labels and concatenate to build a cohort table)",
    )
    benchmark.add_argument(
        "--flowcell", default="fc1",
        help="flow-cell id recorded as provenance in proteoform_abundances.tsv",
    )
    benchmark.add_argument(
        "--lane", default="1",
        help="lane within the flow cell (an experiment has 4-12 lanes, one per sample)",
    )
    benchmark.add_argument("--max-iter", type=int, default=300)
    benchmark.add_argument("--tol", type=float, default=1e-7)
    benchmark.add_argument("--block-size", type=int, default=4096)
    benchmark.add_argument(
        "--save-traces",
        action="store_true",
        help="also save the generated molecule-by-cycle matrix and simulation truth",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark-tau":
        result = run_tau_like_benchmark(
            n_molecules=args.molecules,
            n_active=args.active,
            concentration=args.concentration,
            repeats=args.repeats,
            missing_rate=args.missing_rate,
            seed=args.seed,
            max_iter=args.max_iter,
            tol=args.tol,
            block_size=args.block_size,
            return_responsibilities=False,
        )
        configuration = {
            "command": "benchmark-tau",
            "molecules": args.molecules,
            "active": args.active,
            "concentration": args.concentration,
            "repeats": args.repeats,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "trace_seed": args.seed + 1,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "block_size": args.block_size,
            "sample": args.sample,
            "flowcell": args.flowcell,
            "lane": args.lane,
            "return_responsibilities": False,
        }
        _write_benchmark(
            result,
            args.output,
            save_traces=args.save_traces,
            configuration=configuration,
            sample=args.sample,
            flowcell=args.flowcell,
            lane=args.lane,
        )
        printed = {
            "metrics_valid": result.metrics_valid,
            "metrics": result.metrics if result.metrics_valid else None,
        }
        print(json.dumps(printed, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
