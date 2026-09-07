#!/usr/bin/env python3
"""Shared provenance helpers for reproducible benchmark archives.

The helpers are deliberately dependency-free so they can be used before a
benchmark creates any output.  A run snapshot therefore describes the code
and Git state that actually launched the computation, rather than the state
after archive files have been written.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable


CHECKSUM_MANIFEST = "checksums.sha256"
RUN_STATUS = "run_status.json"


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with an explicit ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def capture_git_state(repo_root: Path) -> dict[str, Any]:
    """Capture the commit and dirty state without mutating the repository."""

    inside = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "git_available": False,
            "git_commit": None,
            "git_dirty": None,
            "git_status_porcelain": None,
        }

    commit_result = _run_git(repo_root, "rev-parse", "HEAD")
    status_result = _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status_result.returncode != 0:
        status_lines: list[str] | None = None
        dirty: bool | None = None
    else:
        status_lines = status_result.stdout.splitlines()
        dirty = bool(status_lines)
    return {
        "git_available": True,
        "git_commit": (
            commit_result.stdout.strip() if commit_result.returncode == 0 else None
        ),
        "git_dirty": dirty,
        "git_status_porcelain": status_lines,
    }


def _archive_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def capture_run_provenance(
    *,
    repo_root: Path,
    command: Iterable[str],
    source_paths: Iterable[Path],
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Capture immutable run-start provenance before any archive output exists."""

    started_utc = utc_now()
    git_state = capture_git_state(repo_root)
    resolved_sources = sorted(
        {Path(path).resolve() for path in source_paths},
        key=lambda path: _archive_path(path, repo_root),
    )
    missing_sources = [
        _archive_path(path, repo_root)
        for path in resolved_sources
        if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "cannot capture provenance; missing source files: "
            + ", ".join(missing_sources)
        )

    lock_path = (repo_root / "uv.lock") if lock_path is None else lock_path
    lock_exists = lock_path.is_file()
    return {
        "created_utc": started_utc,
        "provenance_captured_before_outputs": True,
        "command": [str(item) for item in command],
        **git_state,
        "uv_lock_path": _archive_path(lock_path, repo_root),
        "uv_lock_sha256": sha256_file(lock_path) if lock_exists else None,
        "source_sha256": {
            _archive_path(path, repo_root): sha256_file(path)
            for path in resolved_sources
        },
        "output_checksums": {
            "algorithm": "sha256",
            "path": CHECKSUM_MANIFEST,
            "self_excluded": True,
        },
    }


def write_run_status(
    output_dir: Path,
    provenance: dict[str, Any],
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    """Write an explicit running, success, or failure record."""

    if status not in {"running", "success", "failed"}:
        raise ValueError(f"unsupported archive status: {status}")
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "started_utc": provenance["created_utc"],
        "completed_utc": None if status == "running" else utc_now(),
        "git_commit_at_start": provenance["git_commit"],
        "git_dirty_at_start": provenance["git_dirty"],
    }
    if error is not None:
        record["failure"] = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        record["provenance_at_start"] = provenance
    (output_dir / RUN_STATUS).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_archive_checksums(output_dir: Path) -> Path:
    """Hash every regular archive file except the checksum manifest itself."""

    manifest_path = output_dir / CHECKSUM_MANIFEST
    files = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in files
    ]
    manifest_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return manifest_path


def finish_archive(
    output_dir: Path,
    provenance: dict[str, Any],
    *,
    error: BaseException | None = None,
) -> None:
    """Finalize status and checksums for a successful or failed run."""

    write_run_status(
        output_dir,
        provenance,
        status="failed" if error is not None else "success",
        error=error,
    )
    write_archive_checksums(output_dir)


def begin_archive(output_dir: Path, provenance: dict[str, Any]) -> None:
    """Create the output directory only after provenance has been captured."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_status(output_dir, provenance, status="running")


def benchmark_source_paths(repo_root: Path, runner_path: Path) -> list[Path]:
    """Return the runner, provenance helper, package metadata, and package code."""

    return [
        runner_path.resolve(),
        Path(__file__).resolve(),
        repo_root / "pyproject.toml",
        *sorted((repo_root / "src" / "proteoem").glob("*.py")),
    ]


def run_archive_main(
    *,
    output_dir: Path,
    provenance: dict[str, Any],
    function,
) -> int:
    """Run one archive body and leave an explicit status on exceptions."""

    try:
        begin_archive(output_dir, provenance)
        function()
    except Exception as error:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            finish_archive(output_dir, provenance, error=error)
        except Exception as finalize_error:  # pragma: no cover - filesystem failure
            print(
                "failed to write benchmark failure provenance: "
                f"{type(finalize_error).__name__}: {finalize_error}",
                file=sys.stderr,
            )
        raise
    finish_archive(output_dir, provenance)
    return 0
