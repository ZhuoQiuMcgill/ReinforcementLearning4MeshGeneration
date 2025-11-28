"""Custom callbacks for the v2 mesh RL training pipeline.

This module currently provides :class:`MeshEvalCallback`, which wraps
Stable-Baselines3's :class:`EvalCallback` so that we mirror the legacy
pipeline behaviour for model saving:

* periodic evaluation on a separate environment
* saving the best model so far under ``best_model``

Numbered checkpoints for each evaluation step are intentionally not
kept in v2 to keep the on-disk footprint smaller; callers can still
rely on the final "mesh" model and the best model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class MeshEvalCallback(EvalCallback):
    """Evaluation callback that saves a "best_model" during training.

    This is a thin wrapper around :class:`EvalCallback` that configures
    the evaluation frequency and output directories. It relies entirely
    on the parent implementation to:

    * periodically evaluate the current policy on ``eval_env``
    * save the best model so far to ``best_model_dir / "best_model"``
    * write ``evaluations.npz`` with evaluation logs
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        log_dir: Path,
        best_model_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        render: bool,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=str(best_model_dir),
            # ``EvalCallback`` will create ``evaluations.npz`` inside this
            # directory; we keep the default naming.
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
