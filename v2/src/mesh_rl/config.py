"""Configuration objects for the v2 mesh RL pipeline.

This module defines simple, serializable configuration structures for
training and evaluation. It is intentionally independent of any
specific RL library so that tests and tooling can construct configs
without importing heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


DeviceChoice = Literal["auto", "cpu", "cuda"]


@dataclass
class RLConfig:
    """High-level configuration for a single training run.

    This is the v2 representation of the global settings that were
    previously scattered across module-level variables in
    ``rl/baselines/RL_Mesh.py``. All algorithm-specific hyperparameters
    that used to be hard-coded in the legacy script (e.g. SAC learning
    rate, batch size, net architectures) are exposed via
    :attr:`algo_kwargs` so that they can be overridden without modifying
    library code.
    """

    algo: str
    domain: str
    total_timesteps: int
    seed: int
    version: str
    device: DeviceChoice = "auto"
    # Additional Stable-Baselines3 keyword arguments for the selected
    # algorithm. Defaults correspond to the legacy ``RL_Mesh.py``
    # settings (see ``mesh_rl.algorithms.sb3_algos``), but callers can
    # override any of them here.
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Evaluation settings mirroring the legacy ``CustomCallback``
    # defaults in ``RL_Mesh.mesh_learning``. These control how often we
    # evaluate during training and how we select ``best_model``.
    eval_freq: int = 1000
    eval_episodes: int = 1
    eval_deterministic: bool = False
    eval_render: bool = False


@dataclass
class PathConfig:
    """Resolved filesystem locations for a training or evaluation run.

    All writeable locations must live under ``v2/outputs`` by design.
    ``root_config_path`` may point to the legacy ``config`` file and is
    treated as read-only.
    """

    project_root: Path
    outputs_root: Path
    root_config_path: Optional[Path] = None

    @property
    def logs_root(self) -> Path:
        return self.outputs_root / "logs"

    @property
    def tensorboard_root(self) -> Path:
        return self.outputs_root / "tensorboard"

    @property
    def models_root(self) -> Path:
        return self.outputs_root / "models"

    @property
    def evaluation_root(self) -> Path:
        return self.outputs_root / "evaluation"


def make_default_paths(project_root: Path) -> PathConfig:
    """Construct a default ``PathConfig`` rooted at ``v2/outputs``.

    The caller is responsible for creating the directories on disk when
    needed; this function only computes paths.
    """

    outputs_root = project_root / "v2" / "outputs"
    root_config = project_root / "config"
    return PathConfig(
        project_root=project_root,
        outputs_root=outputs_root,
        root_config_path=root_config if root_config.is_file() else None,
    )


def resolve_device(preferred: DeviceChoice = "auto") -> str:
    """Resolve the actual device string to use for training.

    This helper deliberately returns a *string*; the actual torch
    ``device`` object should be constructed inside the training
    integration layer so that this module does not depend on torch.
    """

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda"

    # auto: prefer CUDA if available
    try:  # pragma: no cover - behaviour depends on runtime environment
        import torch  # type: ignore[import]

        if torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
    except Exception:  # pragma: no cover - best-effort fallback
        pass

    return "cpu"
