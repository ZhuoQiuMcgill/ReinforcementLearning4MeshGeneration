"""Training loop primitives for the v2 mesh RL pipeline.

This module provides a minimal single-environment training function
that mirrors the high-level behaviour of the legacy ``mesh_learning``
function, but expressed as a reusable library API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback

from ..algorithms import build_model
from ..config import RLConfig, PathConfig, make_default_paths
from ..envs.boundary_env import BoudaryEnv
from ..paths import algo_log_dir, algo_model_path, algo_tensorboard_dir, resolve_domain_path
from .callbacks import MeshEvalCallback


def train_single_env(
    cfg: RLConfig,
    paths: Optional[PathConfig] = None,
    *,
    project_root: Optional[Path] = None,
    model_path: Optional[Path] = None,
    callback: Optional[BaseCallback] = None,
    stage_index: Optional[int] = None,
) -> Path:
    """Train a single environment according to ``cfg`` and return model path.

    Parameters
    ----------
    cfg:
        High-level training configuration (algo, domain, timesteps, etc.).
    paths:
        Precomputed path configuration. If omitted, ``project_root`` must
        be provided and a default ``PathConfig`` rooted at ``v2/outputs``
        will be constructed.
    project_root:
        Project root directory; used only when ``paths`` is not given.
    model_path:
        Optional existing model checkpoint to load and continue
        training from.
    callback:
        Optional SB3 callback to use during training.

    Returns
    -------
    Path
        Filesystem path to the saved model checkpoint.
    """

    if paths is None:
        if project_root is None:
            raise ValueError("Either 'paths' or 'project_root' must be provided.")
        paths = make_default_paths(project_root)

    # Resolve domain path and construct environment. We mirror the
    # legacy pipeline by using a separate evaluation environment with
    # the same domain.
    domain_file = resolve_domain_path(paths, cfg.domain)
    env = (
        BoudaryEnv.from_domain_file(str(domain_file))
        if hasattr(BoudaryEnv, "from_domain_file")
        else BoudaryEnv(domain_file)
    )
    eval_env = (
        BoudaryEnv.from_domain_file(str(domain_file))
        if hasattr(BoudaryEnv, "from_domain_file")
        else BoudaryEnv(domain_file)
    )

    # Determine log / tb / model locations. When ``stage_index`` is
    # provided (curriculum training), we scope outputs under a per-stage
    # subdirectory to mirror the legacy layout.
    log_dir = algo_log_dir(paths, cfg.algo, cfg.version, stage=stage_index)
    tb_dir = algo_tensorboard_dir(paths, cfg.algo, cfg.version, stage=stage_index)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Build or load model
    if model_path is not None and model_path.is_file():
        from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

        algo_lower = cfg.algo.lower()
        if algo_lower == "a2c":
            model = A2C.load(model_path, env=env)
        elif algo_lower == "ddpg":
            model = DDPG.load(model_path, env=env)
        elif algo_lower == "ppo":
            model = PPO.load(model_path, env=env)
        elif algo_lower == "sac":
            model = SAC.load(model_path, env=env)
        elif algo_lower == "td3":
            model = TD3.load(model_path, env=env)
        else:  # pragma: no cover - guarded by earlier validation
            raise ValueError(f"Unsupported algorithm for loading: {cfg.algo!r}")
    else:
        model = build_model(cfg.algo, env, cfg, tensorboard_log=str(tb_dir))

    # Build an evaluation callback that mirrors the legacy pipeline
    # behaviour:
    #
    # * periodically evaluate on a separate env
    # * save the best model under ``best_model``
    # * save numbered checkpoints at each evaluation
    eval_cb = MeshEvalCallback(
        eval_env=eval_env,
        log_dir=log_dir,
        best_model_dir=log_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=cfg.eval_deterministic,
        render=cfg.eval_render,
        verbose=1,
    )

    callbacks: list[BaseCallback] = []
    if callback is not None:
        callbacks.append(callback)
    callbacks.append(eval_cb)
    callbacks.append(ProgressBarCallback())

    cb: BaseCallback
    if len(callbacks) == 1:
        cb = callbacks[0]
    else:
        cb = CallbackList(callbacks)

    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)

    # Save model
    stage_for_path = 0 if stage_index is None else stage_index
    out_path = algo_model_path(paths, cfg.algo, cfg.version, stage=stage_for_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    return out_path
