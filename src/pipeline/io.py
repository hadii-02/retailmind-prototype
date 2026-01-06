# src/pipeline/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Centralized paths used by the pipeline.
    All paths are relative to repo_root unless explicitly overridden.
    """
    repo_root: Path
    data_dir: Path
    processed_dir: Path
    dashboard_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path, dashboard_out: Path | None = None) -> "Paths":
        data_dir = repo_root / "data"
        processed_dir = data_dir / "processed"
        dashboard_dir = dashboard_out if dashboard_out is not None else (data_dir / "dashboard")
        return Paths(
            repo_root=repo_root,
            data_dir=data_dir,
            processed_dir=processed_dir,
            dashboard_dir=dashboard_dir,
        )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
