"""Curriculum training orchestration for the v2 mesh RL pipeline.

This module re-expresses the high-level curriculum behaviour of the
legacy ``RL_Mesh.curriculum_learning`` function in a reusable form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .train_loop import train_single_env
from ..config import RLConfig, PathConfig, make_default_paths
from ..paths import algo_model_path


@dataclass
class CurriculumStage:
    """Single curriculum stage: domain + timesteps."""

    index: int
    domain: str
    timesteps: int


def default_curriculum(algo: str) -> List[CurriculumStage]:
    """Return the default curriculum stages for a given algorithm.

    This mirrors the currently active behaviour in the legacy script,
    which effectively trains on a single "random1_1" domain for a
    fixed number of steps, while leaving other potential stages
    commented out.
    """

    # Legacy RL_Mesh currently uses a single active stage with 1.5M steps
    # on the "random1_1" domain. Additional stages can be added later as
    # needed by extending this list.
    return [CurriculumStage(index=0, domain="random1_1", timesteps=1_500_000)]


def train_curriculum(
    cfg: RLConfig,
    paths: Optional[PathConfig] = None,
    *,
    project_root: Optional[Path] = None,
    stages: Optional[List[CurriculumStage]] = None,
) -> Path:
    """Train a curriculum of stages and return the final model path.

    The curriculum definition comes from ``default_curriculum`` for now
    and may be generalized in future revisions.
    """

    if paths is None:
        if project_root is None:
            raise ValueError("Either 'paths' or 'project_root' must be provided.")
        paths = make_default_paths(project_root)

    # Allow callers (e.g. tests or custom CLIs) to provide an explicit
    # curriculum. When omitted, we fall back to the legacy-style default.
    if stages is None:
        stages = default_curriculum(cfg.algo)
    last_model: Optional[Path] = None

    for stage in stages:
        # Stage-specific config. We propagate algorithm hyperparameters and
        # evaluation settings so that all stages share the same RL setup,
        # differing only in domain and timesteps.
        stage_cfg = RLConfig(
            algo=cfg.algo,
            domain=stage.domain,
            total_timesteps=stage.timesteps,
            seed=cfg.seed,
            version=cfg.version,
            device=cfg.device,
            algo_kwargs=cfg.algo_kwargs,
            eval_freq=cfg.eval_freq,
            eval_episodes=cfg.eval_episodes,
            eval_deterministic=cfg.eval_deterministic,
            eval_render=cfg.eval_render,
        )

        # Previous stage model (if any)
        model_path = last_model if last_model is not None else None

        # Train a single stage
        out_path = train_single_env(
            stage_cfg,
            paths=paths,
            model_path=model_path,
            stage_index=stage.index,
        )

        # For compatibility with the legacy layout, we consider the
        # curriculum stage index when computing the model path; the
        # simple single-stage case uses stage 0.
        last_model = algo_model_path(paths, cfg.algo, cfg.version, stage.index)
        if out_path != last_model:
            # Keep the recorded path consistent with the helper.
            last_model = out_path

    if last_model is None:
        raise RuntimeError("Curriculum contained no stages.")

    return last_model
