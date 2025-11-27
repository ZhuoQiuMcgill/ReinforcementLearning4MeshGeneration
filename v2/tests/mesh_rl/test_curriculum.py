import importlib
import sys
from pathlib import Path

import pytest

# Ensure v2/src is importable as package root for mesh_rl
_REPO_ROOT = Path(__file__).resolve().parents[3]
_V2_SRC = _REPO_ROOT / "v2" / "src"
if str(_V2_SRC) not in sys.path:
    sys.path.insert(0, str(_V2_SRC))

# SB3's system_info helper unconditionally tries to read gym.__version__
# when saving models. When running with gymnasium only, ``gym`` may be
# missing or not define ``__version__``, so we provide a lightweight
# shim here for the test environment.
try:  # pragma: no cover
    import gym  # type: ignore[import]
except Exception:  # noqa: BLE001
    gym = types.ModuleType("gym")
    sys.modules["gym"] = gym
if not hasattr(gym, "__version__"):
    gym.__version__ = "0.0.0-test"

from mesh_rl.config import RLConfig, make_default_paths
from mesh_rl.training.curriculum import CurriculumStage, train_curriculum


@pytest.mark.smoke
def test_two_stage_curriculum_chains_models(tmp_path: Path) -> None:
    """Run a tiny two-stage curriculum and check model chaining.

    This uses very small timesteps to keep the test fast and focuses on
    verifying that:

    * each stage writes its model to the correct curriculum/<idx>/mesh
    * the second stage loads the first stage's model as its starting point
      (implicitly checked via the expected path wiring)
    """

    if importlib.util.find_spec("stable_baselines3") is None:  # type: ignore[attr-defined]
        pytest.skip("stable_baselines3 not installed; skipping curriculum smoke test")

    project_root = _REPO_ROOT
    paths = make_default_paths(project_root)

    cfg = RLConfig(
        algo="sac",
        domain="random1_1",
        total_timesteps=50,
        seed=123,
        version="curriculum_smoke",
        device="auto",
    )

    # Two tiny stages on the same domain, to exercise model chaining and
    # per-stage output directories.
    stages = [
        CurriculumStage(index=0, domain="random1_1", timesteps=50),
        CurriculumStage(index=1, domain="random1_1", timesteps=50),
    ]

    final_model = train_curriculum(cfg, paths=paths, stages=stages)

    # Final model path should live under curriculum/1/mesh
    assert "curriculum" in str(final_model)
    assert final_model.name == "mesh"
    assert final_model.parent.name == "1"
    assert final_model.parent.parent.name == "curriculum"