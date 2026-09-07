#!/usr/bin/env python3
"""Report exact trace-class counts for the seeded tau-like simulation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from proteoem import (
    aggregate_probe_counts,
    build_emission_matrix,
    compress_probe_count_classes,
)
from proteoem.benchmark import (
    BASE_PROBE_LABELS,
    default_tau_logical_probe_rates,
    default_tau_probe_rates,
    make_tau_like_panel,
    sparse_tau_weights,
)
from proteoem.simulate import simulate_traces


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize(
    counts: np.ndarray,
    *,
    n_molecules: int,
    representation: str,
    n_cycles: int,
    n_logical_probes: int,
    n_candidates: int,
    missing_fraction: float,
    retained_logical_probes: int,
) -> dict[str, int | float | str]:
    repeated = counts > 1
    n_classes = int(counts.size)
    return {
        "n_molecules": n_molecules,
        "n_cycles": n_cycles,
        "n_logical_probes": n_logical_probes,
        "n_candidates": n_candidates,
        "missing_fraction": missing_fraction,
        "representation": representation,
        "retained_logical_probes": retained_logical_probes,
        "n_classes": n_classes,
        "singleton_classes": int(np.sum(counts == 1)),
        "repeated_classes": int(np.sum(repeated)),
        "molecules_in_repeated_classes": int(np.sum(counts[repeated])),
        "repeated_rows_beyond_representative": n_molecules - n_classes,
        "max_class_size": int(np.max(counts)),
        "compression_n_over_t": n_molecules / n_classes,
        "dense_entries_per_em_iteration": n_classes * n_candidates,
    }


def audit_size(
    n_molecules: int, *, missing_rate: float, mixture_seed: int, trace_seed: int
) -> list[dict[str, int | float | str]]:
    panel = make_tau_like_panel(repeats=3)
    alpha, beta = default_tau_probe_rates(panel)
    logical_alpha, logical_beta = default_tau_logical_probe_rates(panel)
    logical_q = build_emission_matrix(
        panel.logical_profiles,
        alpha=logical_alpha,
        beta=logical_beta,
    )
    weights = sparse_tau_weights(panel, n_active=32, seed=mixture_seed)
    simulation = simulate_traces(
        panel.profiles,
        n_molecules,
        weights=weights,
        alpha=alpha,
        beta=beta,
        missing_rate=missing_rate,
        seed=trace_seed,
    )

    raw_aggregated = simulation.aggregated
    raw_counts = raw_aggregated.counts
    n_logical = len(BASE_PROBE_LABELS)
    sufficient = aggregate_probe_counts(
        raw_aggregated,
        panel.cycle_to_probe,
        n_logical_probes=n_logical,
    )
    relative = compress_probe_count_classes(sufficient, logical_q)

    common = {
        "n_molecules": n_molecules,
        "n_cycles": panel.n_probes,
        "n_logical_probes": n_logical,
        "n_candidates": panel.n_candidates,
        "missing_fraction": raw_aggregated.missing_fraction,
    }
    return [
        _summarize(
            raw_counts,
            representation="ordered_trace_and_mask",
            retained_logical_probes=n_logical,
            **common,
        ),
        _summarize(
            sufficient.counts,
            representation="exchangeable_repeat_sufficient_counts",
            retained_logical_probes=n_logical,
            **common,
        ),
        _summarize(
            relative.counts,
            representation="relative_trace_likelihood_counts",
            retained_logical_probes=relative.n_retained_probes,
            **common,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--molecules", type=int, nargs="+", default=[5_000, 200_000, 1_000_000]
    )
    parser.add_argument("--missing-rate", type=float, default=0.02)
    parser.add_argument("--mixture-seed", type=int, default=7)
    parser.add_argument("--trace-seed", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = _git_state(repo_root)
    started_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, int | float | str]] = []
    failures: list[dict[str, int | str]] = []
    for n_molecules in args.molecules:
        try:
            rows.extend(
                audit_size(
                    n_molecules,
                    missing_rate=args.missing_rate,
                    mixture_seed=args.mixture_seed,
                    trace_seed=args.trace_seed,
                )
            )
        except Exception as exc:  # preserve an auditable partial archive
            failures.append(
                {
                    "n_molecules": n_molecules,
                    "status": "exception",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )

    if not rows:
        raise RuntimeError("No scale-audit rows were generated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    archive_root = args.output.parent
    configuration_path = archive_root / "configuration.json"
    environment_path = archive_root / "environment.json"
    failures_path = archive_root / "failures.tsv"
    checksums_path = archive_root / "output_checksums.sha256"

    _write_json(
        configuration_path,
        {
            "schema_version": 1,
            "claim_boundary": (
                "Synthetic trace-class arithmetic audit only; not a fitted runtime "
                "benchmark and not evidence for experimental platform scale."
            ),
            "molecules": args.molecules,
            "missing_rate": args.missing_rate,
            "mixture_seed": args.mixture_seed,
            "trace_seed": args.trace_seed,
            "representations": [
                "ordered_trace_and_mask",
                "exchangeable_repeat_sufficient_counts",
                "relative_trace_likelihood_counts",
            ],
        },
    )

    source_paths = [
        Path("scripts/audit_trace_class_scale.py"),
        Path("src/proteoem/benchmark.py"),
        Path("src/proteoem/em.py"),
        Path("src/proteoem/emissions.py"),
        Path("src/proteoem/equivalence.py"),
        Path("src/proteoem/simulate.py"),
    ]
    lock_path = repo_root / "uv.lock"
    _write_json(
        environment_path,
        {
            "schema_version": 1,
            "creation_time_utc": started_utc,
            "command": sys.argv,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "dependency_lock_sha256": _sha256(lock_path) if lock_path.exists() else None,
            "source_sha256": {
                str(path): _sha256(repo_root / path) for path in source_paths
            },
        },
    )

    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["n_molecules", "status", "exception_type", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(failures)

    checksum_targets = [
        args.output,
        configuration_path,
        environment_path,
        failures_path,
    ]
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n" for path in sorted(checksum_targets)
        ),
        encoding="utf-8",
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
