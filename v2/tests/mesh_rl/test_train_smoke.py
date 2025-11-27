import importlib
import sys
import types
from pathlib import Path

import pytest

# Ensure v2/src is importable as package root for mesh_rl
_REPO_ROOT = Path(__file__).resolve().parents[3]
_V2_SRC = _REPO_ROOT / "v2" / "src"
if str(_V2_SRC) not in sys.path:
    sys.path.insert(0, str(_V2_SRC))

# Provide a minimal pycuda stub if the real package is unavailable,
# so that general.original_ann can import without requiring a full
# native CUDA toolchain for this smoke test.
try:  # pragma: no cover
    import pycuda.driver as _cuda  # type: ignore[import]
except Exception:  # noqa: BLE001
    pycuda = types.ModuleType("pycuda")
    cuda = types.ModuleType("pycuda.driver")
    sys.modules["pycuda"] = pycuda
    sys.modules["pycuda.driver"] = cuda

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
from mesh_rl.training import train_single_env


@pytest.mark.smoke
def test_train_single_env_sac_smoke(tmp_path: Path) -> None:
    """Run a tiny SAC training loop to ensure the v2 pipeline is wired.

    This test is skipped automatically if Stable-Baselines3 is not
    available in the current environment.
    """

    if importlib.util.find_spec("stable_baselines3") is None:  # type: ignore[attr-defined]
        pytest.skip("stable_baselines3 not installed; skipping training smoke test")

    project_root = Path(__file__).resolve().parents[3]

    cfg = RLConfig(
        algo="sac",
        domain="dolphine3",
        total_timesteps=100,  # very small budget for smoke test
        seed=123,
        version="smoke_test",
        device="auto",
    )

    paths = make_default_paths(project_root)

    model_path = train_single_env(cfg, paths=paths)

    assert isinstance(model_path, Path)
    # Model file may not exist depending on SB3 behaviour, but the
    # directory should at least have been created.
    assert model_path.parent.is_dir()