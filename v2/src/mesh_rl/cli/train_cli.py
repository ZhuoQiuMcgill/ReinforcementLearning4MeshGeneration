"""Command-line entrypoint for training the v2 mesh RL pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config import RLConfig, make_default_paths, resolve_device
from ..training.curriculum import CurriculumStage, train_curriculum


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML training config file if it exists."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level.")
    return data


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train mesh RL agent (v2)")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file for training params")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm name (e.g. sac, ppo, a2c)")
    parser.add_argument(
        "--domain",
        type=str,
        default="dolphine3",
        help="Domain key corresponding to a JSON file under v2/data/domains",
    )
    parser.add_argument("--steps", type=int, default=1_500_000, help="Total timesteps (per curriculum stage)")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    parser.add_argument("--version", type=str, default="v2_run", help="Experiment version tag")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device preference")

    args = parser.parse_args(argv)

    # This file lives at v2/src/mesh_rl/cli/train_cli.py
    # parents[3] -> .../v2, parents[4] -> repo root.
    project_root = Path(__file__).resolve().parents[4]

    # Optionally load a YAML config and let explicit CLI flags override it.
    cfg_data: Dict[str, Any] = {}
    if args.config is not None:
        cfg_path = Path(args.config)
        cfg_data.update(_load_config(cfg_path))

    # CLI args take precedence over YAML file values.
    cfg_data["algo"] = args.algo or cfg_data.get("algo", "sac")
    cfg_data["domain"] = args.domain or cfg_data.get("domain", "dolphine3")
    cfg_data["total_timesteps"] = args.steps or cfg_data.get("total_timesteps", 1_500_000)
    cfg_data["seed"] = args.seed or cfg_data.get("seed", 999)
    cfg_data["version"] = args.version or cfg_data.get("version", "v2_run")
    cfg_data["device"] = args.device or cfg_data.get("device", "auto")

    algo_kwargs = cfg_data.get("algo_kwargs", {})
    if not isinstance(algo_kwargs, dict):
        raise ValueError("'algo_kwargs' in YAML config must be a mapping")

    cfg = RLConfig(
        algo=str(cfg_data["algo"]),
        domain=str(cfg_data["domain"]),
        total_timesteps=int(cfg_data["total_timesteps"]),
        seed=int(cfg_data["seed"]),
        version=str(cfg_data["version"]),
        device=str(cfg_data["device"]),  # type: ignore[arg-type]
        algo_kwargs=algo_kwargs,
    )

    # Resolve paths and device up-front so we can display them to the user
    # before training starts.
    paths = make_default_paths(project_root)
    resolved_device = resolve_device(cfg.device)

    log_dir = paths.logs_root / cfg.algo / cfg.version / "curriculum" / "0"
    model_dir = log_dir

    print("[mesh_rl] Training configuration:")
    print(f"  algo   : {cfg.algo}")
    print(f"  domain : {cfg.domain} (v2/data/domains/{cfg.domain}.json)")
    print(f"  steps  : {cfg.total_timesteps}")
    print(f"  seed   : {cfg.seed}")
    print(f"  device : {resolved_device}")
    print(f"  logs   : {log_dir}")
    print(f"  model  : {model_dir / 'mesh'}")

    # By default the CLI runs a single-stage curriculum whose domain and
    # timesteps come from the CLI flags, while the library default
    # (``stages=None``) mirrors the legacy hard-coded curriculum.
    stages = [CurriculumStage(index=0, domain=cfg.domain, timesteps=cfg.total_timesteps)]

    model_path = train_curriculum(cfg, paths=paths, stages=stages)
    print(f"[mesh_rl] Training complete. Final model saved at: {model_path}")


if __name__ == "__main__":  # pragma: no cover
    main()