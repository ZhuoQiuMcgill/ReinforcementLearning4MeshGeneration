"""Evaluation loop for the v2 mesh RL pipeline.

This module captures the high-level behaviour of ``rl/baselines/testbed``
so that models trained with v2 code can be evaluated in a consistent,
script-independent way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from ..config import PathConfig, make_default_paths
from ..envs.boundary_env import BoudaryEnv
from ..paths import evaluation_output_dir, resolve_domain_path


@dataclass
class EvalConfig:
    """Configuration for evaluating one or more models on one or more domains."""

    algo: str
    model_paths: List[Path]
    domains: List[str]
    version: str
    deterministic: bool = False
    render: bool = False


def _load_model(algo: str, model_path: Path, env: BoudaryEnv):
    algo_lower = algo.lower()
    if algo_lower == "a2c":
        return A2C.load(model_path, env=env)
    if algo_lower == "ddpg":
        return DDPG.load(model_path, env=env)
    if algo_lower == "ppo":
        return PPO.load(model_path, env=env)
    if algo_lower == "sac":
        return SAC.load(model_path, env=env)
    if algo_lower == "td3":
        return TD3.load(model_path, env=env)
    raise ValueError(f"Unsupported algorithm: {algo!r}")


def evaluate_models(
    cfg: EvalConfig,
    paths: Optional[PathConfig] = None,
    *,
    project_root: Optional[Path] = None,
    save_summary: bool = True,
) -> Dict[str, Dict[str, List[int]]]:
    """Evaluate one or more models on configured domains.

    Returns a nested dictionary of summary statistics keyed by
    model-id, broadly mirroring the structure of the legacy
    ``evaluation.txt`` output.
    """

    if paths is None:
        if project_root is None:
            raise ValueError("Either 'paths' or 'project_root' must be provided.")
        paths = make_default_paths(project_root)

    out_dir = evaluation_output_dir(paths, cfg.version)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare environments for each domain
    envs: List[BoudaryEnv] = []
    for idx, domain_key in enumerate(cfg.domains):
        domain_file = resolve_domain_path(paths, domain_key)
        env = BoudaryEnv.from_domain_file(str(domain_file)) if hasattr(BoudaryEnv, "from_domain_file") else BoudaryEnv(domain_file)
        envs.append(env)

    results: Dict[str, Dict[str, List[int]]] = {}

    for model_path in cfg.model_paths:
        model_id = model_path.stem
        results[model_id] = {
            "completed": [],
            "n_elements": [],
        }

        for env in envs:
            model = _load_model(cfg.algo, model_path, env)
            obs, _info = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=cfg.deterministic)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, rewards, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:  # pragma: no cover - legacy-style 4-tuple fallback
                    obs, rewards, done, info = step_out
                if cfg.render:
                    env.render()
                if done:
                    break

            results[model_id]["completed"].append(int(bool(info.get("is_complete", False))))
            results[model_id]["n_elements"].append(len(env.generated_meshes))
            env.close()

    if save_summary:
        import json

        summary_path = out_dir / "evaluation_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)

    return results
