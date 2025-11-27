"""Path helpers for the v2 mesh RL pipeline.

All writeable paths are rooted under ``v2/outputs`` via ``PathConfig``.

Domain files are resolved *only* from the v2-local data directory
``v2/data/domains`` so that the v2 project is completely independent
from the legacy root ``config`` and ``ui/domains`` layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PathConfig


def domain_data_root(paths: PathConfig) -> Path:
    """Return the v2-local root directory for domain JSON files."""

    return paths.project_root / "v2" / "data" / "domains"


def algo_log_dir(paths: PathConfig, algo: str, version: str, stage: Optional[int] = None) -> Path:
    """Return the directory where logs for a given algo/version/stage live."""

    base = paths.logs_root / algo / version
    if stage is not None:
        return base / "curriculum" / str(stage)
    return base


def algo_tensorboard_dir(paths: PathConfig, algo: str, version: str, stage: Optional[int] = None) -> Path:
    """Return the directory where tensorboard runs for a given stage live."""

    base = paths.tensorboard_root / algo / version
    if stage is not None:
        return base / "curriculum" / str(stage)
    return base


def algo_model_path(paths: PathConfig, algo: str, version: str, stage: int) -> Path:
    """Return the expected model checkpoint path for a curriculum stage."""

    return algo_log_dir(paths, algo, version, stage) / "mesh"


def resolve_domain_path(paths: PathConfig, domain_key: str) -> Path:
    """Resolve a domain path by key using v2-local data only.

    Domain JSON files are expected to live under ``v2/data/domains`` as
    ``<domain_key>.json``. If the file does not exist, a
    :class:`FileNotFoundError` is raised.
    """

    candidate = domain_data_root(paths) / f"{domain_key}.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Domain file not found for key {domain_key!r} at {candidate}"
        )
    return candidate
def evaluation_output_dir(paths: PathConfig, version: str) -> Path:
    """Directory where evaluation summaries and figures are stored."""

    return paths.evaluation_root / version
