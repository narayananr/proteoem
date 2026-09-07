#!/usr/bin/env python3
"""Run reproducible multi-seed ProteoEM correctness and scaling benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from archive_provenance import (
    benchmark_source_paths,
    capture_run_provenance,
    run_archive_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteoem import (  # noqa: E402
    abundance_metrics,
    aggregate_probe_counts,
    alignment_profile_log_likelihoods,
    build_emission_matrix,
    candidate_log_likelihoods,
    compress_probe_count_classes,
    default_tau_logical_probe_rates,
    default_tau_probe_rates,
    fit_likelihood_em,
    hard_assignment_counts,
    make_tau_like_panel,
    simulate_traces,
    sparse_tau_weights,
)
from proteoem.emissions import log_likelihood_from_probe_counts  # noqa: E402


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _class_statistics(prefix: str, counts: np.ndarray) -> dict[str, int | float]:
    counts = np.asarray(counts, dtype=np.int64)
    repeated = counts > 1
    n_observations = int(counts.sum())
    n_classes = int(counts.size)
    quantiles = np.quantile(counts, [0.5, 0.9, 0.95, 0.99, 0.999])
    return {
        f"{prefix}_classes": n_classes,
        f"{prefix}_singleton_classes": int(np.sum(counts == 1)),
        f"{prefix}_repeated_classes": int(np.sum(repeated)),
        f"{prefix}_molecules_in_repeated_classes": int(np.sum(counts[repeated])),
        f"{prefix}_duplicate_copies": n_observations - n_classes,
        f"{prefix}_compression": n_observations / n_classes,
        f"{prefix}_class_p50": float(quantiles[0]),
        f"{prefix}_class_p90": float(quantiles[1]),
        f"{prefix}_class_p95": float(quantiles[2]),
        f"{prefix}_class_p99": float(quantiles[3]),
        f"{prefix}_class_p999": float(quantiles[4]),
        f"{prefix}_max_class": int(np.max(counts)),
    }


def _timed(function, *args, **kwargs):
    start = time.perf_counter_ns()
    value = function(*args, **kwargs)
    seconds = (time.perf_counter_ns() - start) / 1e9
    return value, seconds


def _worker(
    *,
    n_molecules: int,
    seed: int,
    n_active: int,
    missing_rate: float,
    max_iter: int,
    tol: float,
    block_size: int,
    validate_molecule_fit: bool,
) -> dict[str, Any]:
    panel = make_tau_like_panel(repeats=3)
    physical_alpha, physical_beta = default_tau_probe_rates(panel)
    logical_alpha, logical_beta = default_tau_logical_probe_rates(panel)
    logical_q = build_emission_matrix(
        panel.logical_profiles,
        alpha=logical_alpha,
        beta=logical_beta,
    )
    truth = sparse_tau_weights(panel, n_active=n_active, seed=seed)

    simulation, simulation_seconds = _timed(
        simulate_traces,
        panel.profiles,
        n_molecules,
        weights=truth,
        alpha=physical_alpha,
        beta=physical_beta,
        missing_rate=missing_rate,
        seed=seed + 1,
    )
    raw, raw_grouping_seconds = _timed(lambda: simulation.aggregated)
    sufficient, sufficient_grouping_seconds = _timed(
        aggregate_probe_counts,
        raw,
        panel.cycle_to_probe,
        n_logical_probes=panel.n_logical_probes,
    )
    relative, relative_grouping_seconds = _timed(
        compress_probe_count_classes, sufficient, logical_q
    )
    relative_logs, likelihood_seconds = _timed(
        log_likelihood_from_probe_counts, relative, logical_q
    )
    weighted_fit, weighted_em_seconds = _timed(
        fit_likelihood_em,
        relative_logs,
        counts=relative.counts,
        log_likelihood_offset=relative.log_likelihood_offset,
        max_iter=max_iter,
        tol=tol,
        block_size=block_size,
        return_responsibilities=False,
    )

    hard_counts = hard_assignment_counts(
        relative_logs, counts=relative.counts, split_ties=True
    )
    hard_weights = hard_counts / hard_counts.sum()
    incidence_logs, incidence_likelihood_seconds = _timed(
        alignment_profile_log_likelihoods, raw, panel.profiles, mode="best"
    )
    incidence_fit, incidence_em_seconds = _timed(
        fit_likelihood_em,
        incidence_logs,
        counts=raw.counts,
        max_iter=max_iter,
        tol=tol,
        block_size=block_size,
        return_responsibilities=False,
    )

    metrics_valid = bool(weighted_fit.converged and incidence_fit.converged)
    metrics = {
        "hard": abundance_metrics(hard_weights, truth) if metrics_valid else None,
        "weighted": abundance_metrics(weighted_fit.weights, truth)
        if metrics_valid
        else None,
        "binary": abundance_metrics(incidence_fit.weights, truth)
        if metrics_valid
        else None,
    }
    peak_before_validation = _peak_rss_bytes()
    elapsed_before_validation = (
        simulation_seconds
        + raw_grouping_seconds
        + sufficient_grouping_seconds
        + relative_grouping_seconds
        + likelihood_seconds
        + weighted_em_seconds
        + incidence_likelihood_seconds
        + incidence_em_seconds
    )

    equivalence: dict[str, float | bool] = {"validation_performed": False}
    validation_seconds = 0.0
    if validate_molecule_fit:
        validation_start = time.perf_counter_ns()
        molecule_logs = candidate_log_likelihoods(
            simulation.observations, simulation.Q
        )
        molecule_fit = fit_likelihood_em(
            molecule_logs,
            max_iter=max_iter,
            tol=tol,
            block_size=block_size,
            return_responsibilities=False,
        )
        ordered_logs = candidate_log_likelihoods(raw, simulation.Q)
        ordered_fit = fit_likelihood_em(
            ordered_logs,
            counts=raw.counts,
            max_iter=max_iter,
            tol=tol,
            block_size=block_size,
            return_responsibilities=False,
        )
        validation_seconds = (time.perf_counter_ns() - validation_start) / 1e9
        equivalence = {
            "validation_performed": True,
            "molecule_converged": molecule_fit.converged,
            "ordered_converged": ordered_fit.converged,
            "max_weight_difference_molecule_vs_relative": float(
                np.max(np.abs(molecule_fit.weights - weighted_fit.weights))
            ),
            "max_weight_difference_ordered_vs_relative": float(
                np.max(np.abs(ordered_fit.weights - weighted_fit.weights))
            ),
            "log_likelihood_difference_molecule_vs_relative": float(
                abs(molecule_fit.log_likelihood - weighted_fit.log_likelihood)
            ),
            "log_likelihood_difference_ordered_vs_relative": float(
                abs(ordered_fit.log_likelihood - weighted_fit.log_likelihood)
            ),
        }

    result: dict[str, Any] = {
        "run_id": f"n{n_molecules}_seed{seed}",
        "status": "ok" if metrics_valid else "nonconverged",
        "seed": seed,
        "trace_seed": seed + 1,
        "n_molecules": n_molecules,
        "n_cycles": panel.n_probes,
        "n_logical_probes": panel.n_logical_probes,
        "n_candidates": panel.n_candidates,
        "n_active": n_active,
        "missing_rate_requested": missing_rate,
        "missing_fraction_observed": raw.missing_fraction,
        "retained_assignment_probes": relative.n_retained_probes,
        "likelihood_support_density": weighted_fit.diagnostics[
            "likelihood_support_density"
        ],
        "dense_entries_per_iteration": relative.n_unique * panel.n_candidates,
        "weighted_iterations": weighted_fit.n_iter,
        "weighted_converged": weighted_fit.converged,
        "weighted_terminal_em_residual": weighted_fit.diagnostics[
            "terminal_em_residual"
        ],
        "weighted_log_likelihood_per_molecule": weighted_fit.log_likelihood
        / n_molecules,
        "binary_iterations": incidence_fit.n_iter,
        "binary_converged": incidence_fit.converged,
        "metrics_valid": metrics_valid,
        "simulation_seconds": simulation_seconds,
        "raw_grouping_seconds": raw_grouping_seconds,
        "sufficient_grouping_seconds": sufficient_grouping_seconds,
        "relative_grouping_seconds": relative_grouping_seconds,
        "likelihood_seconds": likelihood_seconds,
        "weighted_em_seconds": weighted_em_seconds,
        "weighted_seconds_per_iteration": weighted_em_seconds
        / max(1, weighted_fit.n_iter),
        "binary_likelihood_seconds": incidence_likelihood_seconds,
        "binary_em_seconds": incidence_em_seconds,
        "timed_total_seconds": elapsed_before_validation,
        "validation_seconds": validation_seconds,
        "peak_rss_before_validation_bytes": peak_before_validation,
        "peak_rss_final_bytes": _peak_rss_bytes(),
        "hard_total_variation": metrics["hard"]["total_variation"]
        if metrics["hard"]
        else None,
        "weighted_total_variation": metrics["weighted"]["total_variation"]
        if metrics["weighted"]
        else None,
        "binary_total_variation": metrics["binary"]["total_variation"]
        if metrics["binary"]
        else None,
        **_class_statistics("raw", raw.counts),
        **_class_statistics("sufficient", sufficient.counts),
        **_class_statistics("relative", relative.counts),
        **equivalence,
        "_weighted_history": weighted_fit.log_likelihood_history.tolist(),
        "_weighted_centered_history": weighted_fit.centered_log_likelihood_history.tolist(),
    }
    return result


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for n_molecules in sorted({int(row["n_molecules"]) for row in rows}):
        selected = [row for row in rows if int(row["n_molecules"]) == n_molecules]
        summary: dict[str, Any] = {
            "n_molecules": n_molecules,
            "n_runs": len(selected),
            "n_valid_runs": sum(bool(row["metrics_valid"]) for row in selected),
        }
        for key in (
            "hard_total_variation",
            "weighted_total_variation",
            "binary_total_variation",
            "weighted_em_seconds",
            "timed_total_seconds",
            "relative_classes",
            "relative_compression",
        ):
            values = [
                float(row[key]) for row in selected if row.get(key) is not None
            ]
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_sd"] = _sd(values)
        summary["peak_rss_max_bytes"] = max(
            int(row["peak_rss_before_validation_bytes"]) for row in selected
        )
        summaries.append(summary)
    return summaries


def _hardware_summary() -> dict[str, Any]:
    """Return reproducibility-relevant hardware fields without device identifiers."""

    summary: dict[str, Any] = {"logical_cpu_count": os.cpu_count()}
    if sys.platform != "darwin":
        return summary
    completed = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return summary
    try:
        record = json.loads(completed.stdout)["SPHardwareDataType"][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return summary
    for source_key, output_key in (
        ("machine_name", "model_name"),
        ("machine_model", "model_identifier"),
        ("chip_type", "chip"),
        ("number_processors", "core_configuration"),
        ("physical_memory", "memory"),
    ):
        if source_key in record:
            summary[output_key] = record[source_key]
    return summary


def _worker_parser(subparsers) -> None:
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--molecules", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--active", type=int, required=True)
    worker.add_argument("--missing-rate", type=float, required=True)
    worker.add_argument("--max-iter", type=int, required=True)
    worker.add_argument("--tol", type=float, required=True)
    worker.add_argument("--block-size", type=int, required=True)
    worker.add_argument("--validate-molecule-fit", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--molecules", type=int, nargs="+", default=[1_000, 5_000])
    run.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    run.add_argument("--active", type=int, default=32)
    run.add_argument("--missing-rate", type=float, default=0.02)
    run.add_argument("--max-iter", type=int, default=300)
    run.add_argument("--tol", type=float, default=1e-7)
    run.add_argument("--block-size", type=int, default=4096)
    run.add_argument("--validation-max-molecules", type=int, default=5_000)
    _worker_parser(subparsers)
    return parser


def _run_benchmark(args, provenance: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    histories = args.output / "likelihood_histories"
    histories.mkdir(exist_ok=True)
    for n_molecules in args.molecules:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--molecules",
                str(n_molecules),
                "--seed",
                str(seed),
                "--active",
                str(args.active),
                "--missing-rate",
                str(args.missing_rate),
                "--max-iter",
                str(args.max_iter),
                "--tol",
                str(args.tol),
                "--block-size",
                str(args.block_size),
            ]
            if n_molecules <= args.validation_max_molecules:
                command.append("--validate-molecule-fit")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC_ROOT)
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark worker failed for N={n_molecules}, seed={seed}:\n"
                    f"{completed.stderr}"
                )
            row = json.loads(completed.stdout)
            raw_history = row.pop("_weighted_history")
            centered_history = row.pop("_weighted_centered_history")
            history_path = histories / f"{row['run_id']}.tsv"
            history_rows = [
                {
                    "iteration": index,
                    "log_likelihood": raw,
                    "centered_log_likelihood": centered,
                }
                for index, (raw, centered) in enumerate(
                    zip(raw_history, centered_history, strict=True)
                )
            ]
            _write_tsv(history_path, history_rows)
            rows.append(row)

    _write_tsv(args.output / "runs.tsv", rows)
    _write_tsv(args.output / "summary.tsv", _aggregate_rows(rows))
    equivalence_rows = [row for row in rows if row["validation_performed"]]
    _write_tsv(
        args.output / "equivalence.tsv",
        [
            {
                key: value
                for key, value in row.items()
                if key == "run_id"
                or key == "n_molecules"
                or key == "seed"
                or "difference_" in key
                or key.endswith("_converged")
            }
            for row in equivalence_rows
        ],
    )
    manifest = {
        **provenance,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hardware": _hardware_summary(),
        "configuration": {
            "molecules": args.molecules,
            "seeds": args.seeds,
            "active": args.active,
            "missing_rate": args.missing_rate,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "block_size": args.block_size,
            "validation_max_molecules": args.validation_max_molecules,
        },
    }
    (args.output / "environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "_worker":
        result = _worker(
            n_molecules=args.molecules,
            seed=args.seed,
            n_active=args.active,
            missing_rate=args.missing_rate,
            max_iter=args.max_iter,
            tol=args.tol,
            block_size=args.block_size,
            validate_molecule_fit=args.validate_molecule_fit,
        )
        print(json.dumps(result, allow_nan=False))
        return 0

    runner_path = Path(__file__).resolve()
    provenance = capture_run_provenance(
        repo_root=REPO_ROOT,
        command=sys.argv,
        source_paths=benchmark_source_paths(REPO_ROOT, runner_path),
    )
    return run_archive_main(
        output_dir=args.output,
        provenance=provenance,
        function=lambda: _run_benchmark(args, provenance),
    )


if __name__ == "__main__":
    raise SystemExit(main())
