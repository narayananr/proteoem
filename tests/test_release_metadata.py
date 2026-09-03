from __future__ import annotations

from importlib.metadata import version as installed_version
import json
from pathlib import Path
import re

import pytest

import proteoem
from proteoem.cli import build_parser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _toml_project_version(path: Path) -> str:
    in_project_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue
        if not in_project_section:
            continue
        match = re.fullmatch(r'version\s*=\s*"([^"]+)"\s*', line)
        if match:
            return match.group(1)
    raise AssertionError("pyproject.toml has no quoted project.version")


def _cff_version(path: Path) -> str:
    matches = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r'version:\s*["\']?([^"\'\s]+)["\']?\s*', raw_line)
        if match:
            matches.append(match.group(1))
    if len(matches) != 1:
        raise AssertionError(
            f"CITATION.cff must contain exactly one top-level version, found {matches}"
        )
    return matches[0]


def test_cli_version_uses_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"proteoem {proteoem.__version__}"


def test_release_versions_are_in_sync() -> None:
    versions = {
        "installed metadata": installed_version("proteoem"),
        "proteoem.__version__": proteoem.__version__,
        "pyproject.toml": _toml_project_version(REPOSITORY_ROOT / "pyproject.toml"),
        "CITATION.cff": _cff_version(REPOSITORY_ROOT / "CITATION.cff"),
    }
    assert len(set(versions.values())) == 1, versions
