import importlib
import json
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
# mirroring the smoke test so that any legacy-dependent imports do not
# require a full native CUDA toolchain in CI.
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
from mesh_rl.evaluation.eval_loop import EvalConfig, evaluate_models
from mesh_rl.training import train_single_env
from mesh_rl.paths import evaluation_output_dir


@pytest.mark.smoke
def test_train_and_evaluate_sac_end_to_end(tmp_path: Path) -> None:
    """Train a tiny SAC model then evaluate it end-to-end.

    This exercises the full v2 pipeline: path resolution, environment
    construction, training with Stable-Baselines3, and evaluation with
    JSON summary emission.

    The test is skipped automatically if Stable-Baselines3 is not
    available in the current environment.
    """

    if importlib.util.find_spec("stable_baselines3") is None:  # type: ignore[attr-defined]
        pytest.skip("stable_baselines3 not installed; skipping train+eval integration test")

    project_root = _REPO_ROOT

    # 1) Train a tiny SAC model on a single domain.
    train_cfg = RLConfig(
        algo="sac",
        domain="dolphine3",
        total_timesteps=100,
        seed=321,
        version="train_eval_integration",
        device="auto",
    )

    paths = make_default_paths(project_root)
    model_path = train_single_env(train_cfg, paths=paths)

    # The SB3 ``save`` implementation typically writes ``model_path`` with
    # a ``.zip`` suffix. We only assert that the parent directory was
    # created successfully, mirroring the smoke test's behaviour.
    assert model_path.parent.is_dir()

    # 2) Evaluate the trained model on the same domain.
    eval_cfg = EvalConfig(
        algo="sac",
        model_paths=[model_path],
        domains=[train_cfg.domain],
        version=train_cfg.version,
        deterministic=False,
        render=False,
    )

    results = evaluate_models(eval_cfg, project_root=project_root, save_summary=True)

    # Basic structure checks on the in-memory results.
    assert isinstance(results, dict)
    assert set(results.keys()) == {model_path.stem}

    model_id = model_path.stem
    completed = results[model_id]["completed"]
    n_elements = results[model_id]["n_elements"]

    assert len(completed) == len(eval_cfg.domains) == 1
    assert len(n_elements) == len(eval_cfg.domains) == 1

    for flag in completed:
        assert isinstance(flag, int)
        assert flag in (0, 1)

    for n in n_elements:
        assert isinstance(n, int)
        assert n >= 0

    # 3) Check that the JSON summary was written to the expected location
    #    and has a compatible structure.
    paths_for_eval = make_default_paths(project_root)
    summary_dir = evaluation_output_dir(paths_for_eval, eval_cfg.version)
    summary_path = summary_dir / "evaluation_summary.json"

    assert summary_path.is_file(), "Evaluation should emit a JSON summary file"

    with summary_path.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert isinstance(summary_data, dict)
    assert set(summary_data.keys()) == set(results.keys())