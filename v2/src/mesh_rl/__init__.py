"""mesh_rl: RL subsystem for quadrilateral mesh generation (v2).

This package lives entirely under ``v2/src`` and must not modify files
outside ``v2/`` at runtime. See ``v2/docs/mesh_rl_reboot_plan.md`` for a
full design overview.

The public API is intentionally small and focuses on library-style
training and evaluation entry points. Subpackages (``envs``,
``training``, ``evaluation``, ``cli``, ``algorithms``, ``legacy``) are
still importable as normal but are considered internal unless re-exported
here.
"""

from .config import PathConfig, RLConfig, make_default_paths
from .envs.boundary_env import BoudaryEnv
from .evaluation.eval_loop import EvalConfig, evaluate_models
from .training.curriculum import CurriculumStage, train_curriculum
from .training.train_loop import train_single_env

__all__ = [
    # Configuration and paths
    "RLConfig",
    "PathConfig",
    "make_default_paths",
    # Training
    "train_single_env",
    "CurriculumStage",
    "train_curriculum",
    # Evaluation
    "EvalConfig",
    "evaluate_models",
    # Environment
    "BoudaryEnv",
]
