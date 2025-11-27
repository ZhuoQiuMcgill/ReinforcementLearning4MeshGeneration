"""Command-line entrypoint for evaluating mesh RL models (v2)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..evaluation import EvalConfig, evaluate_models


def _parse_model_paths(raw: str) -> List[Path]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [Path(p) for p in parts]


def _parse_domains(raw: str) -> List[str]:
    return [d.strip() for d in raw.split(",") if d.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate mesh RL models (v2)")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm name (e.g. sac, ppo, a2c)")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model paths")
    parser.add_argument("--domains", type=str, default="dolphine3", help="Comma-separated list of domain keys")
    parser.add_argument("--version", type=str, default="v2_eval", help="Evaluation version tag")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy during evaluation")
    parser.add_argument("--render", action="store_true", help="Render episodes during evaluation")

    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]

    cfg = EvalConfig(
        algo=args.algo,
        model_paths=_parse_model_paths(args.models),
        domains=_parse_domains(args.domains),
        version=args.version,
        deterministic=args.deterministic,
        render=args.render,
    )

    results = evaluate_models(cfg, project_root=project_root)
    print("Evaluation complete.")
    for model_id, stats in results.items():
        completed = sum(stats["completed"])
        total = len(stats["completed"])
        print(f"  Model {model_id}: completed {completed}/{total} runs")


if __name__ == "__main__":  # pragma: no cover
    main()