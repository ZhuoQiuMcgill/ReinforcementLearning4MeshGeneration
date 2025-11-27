import configparser
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Ensure v2/src is importable as a package root for `mesh_rl`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_V2_SRC = _REPO_ROOT / "v2" / "src"
if str(_V2_SRC) not in sys.path:
    sys.path.insert(0, str(_V2_SRC))

# Lightweight stubs for optional heavy dependencies used by general.original_ann
# and the legacy environment so tests can run without the full RL stack.

# --- gym ---------------------------------------------------------------
try:  # pragma: no cover - only exercised when gym is missing
    import gym  # type: ignore[import]
except Exception:  # noqa: BLE001
    gym = types.ModuleType("gym")

    class _Env:  # minimal base class
        pass

    class _Box:
        def __init__(self, low, high, shape=None, dtype=None):
            if shape is not None:
                shape = tuple(shape)
                self.low = np.full(shape, low, dtype=np.float32)
                self.high = np.full(shape, high, dtype=np.float32)
            else:
                self.low = np.array(low, dtype=np.float32)
                self.high = np.array(high, dtype=np.float32)
                shape = self.low.shape

            self.shape = shape
            self.dtype = dtype or np.float32

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    spaces = types.ModuleType("gym.spaces")
    spaces.Box = _Box  # type: ignore[attr-defined]

    gym.Env = _Env  # type: ignore[attr-defined]
    gym.spaces = spaces  # type: ignore[attr-defined]

    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces

# --- stable_baselines3 -------------------------------------------------
try:  # pragma: no cover - only exercised when SB3 is missing
    import stable_baselines3  # type: ignore[import]
except Exception:  # noqa: BLE001
    sb3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    env_checker = types.ModuleType("stable_baselines3.common.env_checker")

    def _check_env(*_args, **_kwargs):
        return None

    env_checker.check_env = _check_env  # type: ignore[attr-defined]
    common.env_checker = env_checker  # type: ignore[attr-defined]
    sb3.common = common  # type: ignore[attr-defined]

    sys.modules["stable_baselines3"] = sb3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.env_checker"] = env_checker

# --- pygame (for MeshCanvas) ------------------------------------------
try:  # pragma: no cover - only exercised when pygame is missing
    import pygame  # type: ignore[import]
except Exception:  # noqa: BLE001
    pygame = types.ModuleType("pygame")

    class _Surface:
        def __init__(self, size):
            self.size = size

        def fill(self, _color):
            return None

    class _Display:
        def set_mode(self, size):
            return _Surface(size)

        def set_caption(self, _name):
            return None

        def update(self):
            return None

        def quit(self):
            return None

    class _Draw:
        def line(self, _surface, _color, _p1, _p2):
            return None

    class _Transform:
        def flip(self, surface, *_args, **_kwargs):
            return surface

    pygame.Surface = _Surface  # type: ignore[attr-defined]
    pygame.display = _Display()  # type: ignore[attr-defined]
    pygame.draw = _Draw()  # type: ignore[attr-defined]
    pygame.transform = _Transform()  # type: ignore[attr-defined]

    sys.modules["pygame"] = pygame

# --- torch & pycuda (ANN stack) ---------------------------------------
# so that we can import the environment without requiring full Torch/PyCUDA.
try:  # pragma: no cover - only exercised when deps are missing
    import torch  # type: ignore[import]
except Exception:  # noqa: BLE001
    import contextlib

    torch = types.ModuleType("torch")

    def _device(*_args, **_kwargs):
        return "cpu"

    def _set_num_threads(_n: int) -> None:  # noqa: D401
        return None

    torch.device = _device  # type: ignore[attr-defined]
    torch.set_num_threads = _set_num_threads  # type: ignore[attr-defined]
    torch.nn = types.SimpleNamespace(  # type: ignore[attr-defined]
        Sequential=object,
        Linear=object,
        ReLU=object,
        MSELoss=object,
    )
    torch.optim = types.SimpleNamespace(Adam=object)  # type: ignore[attr-defined]
    torch.from_numpy = lambda arr: arr  # type: ignore[attr-defined]
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.save = lambda *_, **__: None  # type: ignore[attr-defined]
    torch.load = lambda *_, **__: None  # type: ignore[attr-defined]
    sys.modules["torch"] = torch

try:  # pragma: no cover - only exercised when deps are missing
    import pycuda.driver as cuda  # type: ignore[import]
except Exception:  # noqa: BLE001
    pycuda = types.ModuleType("pycuda")
    cuda = types.ModuleType("pycuda.driver")
    sys.modules["pycuda"] = pycuda
    sys.modules["pycuda.driver"] = cuda

from mesh_rl.legacy.boundary_env_legacy import BoudaryEnvLegacy as LegacyEnv
from mesh_rl.envs.boundary_env import BoudaryEnv as V2Env
from mesh_rl.geometry import read_polygon


def _project_root() -> Path:
    # v2/tests/mesh_rl -> v2/tests -> v2 -> project root
    return Path(__file__).resolve().parents[3]


def _load_domain(name: str) -> object:
    """Load a domain polygon by name.

    We first try the root `config` file's [domains] section; if that
    path does not exist (e.g. it uses a machine-specific absolute
    path), we fall back to `ui/domains/<name>.json` inside the repo.
    """

    root = _project_root()

    # 1) Try config file if present and the resolved path exists
    cfg_path = root / "config"
    domain_path = None
    if cfg_path.is_file():
        cfg = configparser.ConfigParser()
        read_files = cfg.read(cfg_path)
        if read_files and cfg.has_section("domains") and name in cfg["domains"]:
            candidate = Path(cfg["domains"][name])
            if candidate.is_file():
                domain_path = candidate

    # 2) Fallback to repo-local ui/domains/<name>.json
    if domain_path is None:
        candidate = root / "ui" / "domains" / f"{name}.json"
        if not candidate.is_file():
            raise RuntimeError(
                f"Could not resolve domain '{name}'. Tried config at {cfg_path} "
                f"and repo-local {candidate}."
            )
        domain_path = candidate

    return read_polygon(str(domain_path))


def _make_envs(domain_name: str):
    domain = _load_domain(domain_name)
    legacy = LegacyEnv(domain)
    v2 = V2Env(domain)
    return legacy, v2


def test_reset_and_step_shapes_match():
    """Basic smoke test: v2 env matches legacy env on obs shape and done behaviour.

    For now V2Env is just a re-export of LegacyEnv, so this is mainly a
    guard that we construct environments consistently under v2/.
    """

    legacy, v2 = _make_envs("dolphine3")

    # Seed Python & NumPy for reproducibility
    random.seed(123)
    np.random.seed(123)

    obs_legacy = legacy.reset()
    obs_v2, _info_v2 = v2.reset()

    assert isinstance(obs_legacy, np.ndarray)
    # Gymnasium-style reset already returns just the observation here
    assert isinstance(obs_v2, np.ndarray)
    assert obs_legacy.shape == obs_v2.shape

    # Run a short rollout with random actions sampled from the action space
    for _ in range(10):
        action = v2.action_space.sample()
        o1, r1, d1, info1 = legacy.step(action)
        o2, r2, term2, trunc2, info2 = v2.step(action)
        d2 = bool(term2 or trunc2)

        assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
        assert o1.shape == o2.shape
        # rewards should match exactly for now (same implementation)
        assert math.isfinite(float(r1)) and math.isfinite(float(r2))
        assert math.isclose(float(r1), float(r2), rel_tol=1e-9, abs_tol=1e-9)
        # done flags must match on every step
        assert isinstance(d1, (bool, np.bool_)) and isinstance(d2, (bool, np.bool_))
        assert d1 == d2
        # info dicts should at least share and agree on the is_complete flag
        assert "is_complete" in info1 and "is_complete" in info2
        assert bool(info1["is_complete"]) == bool(info2["is_complete"])

        if d1:
            break


def test_action_space_and_observation_space_consistent():
    legacy, v2 = _make_envs("dolphine3")

    # In the legacy env we may be using a lightweight Box stub; in v2 we
    # use gymnasium.spaces.Box. We only require that both expose ``low`` /
    # ``high`` arrays with the same shape and values, plus identical
    # observation shapes.
    assert hasattr(legacy.action_space, "low") and hasattr(legacy.action_space, "high")
    assert hasattr(v2.action_space, "low") and hasattr(v2.action_space, "high")

    assert np.allclose(np.asarray(legacy.action_space.low), np.asarray(v2.action_space.low))
    assert np.allclose(np.asarray(legacy.action_space.high), np.asarray(v2.action_space.high))
    assert legacy.observation_space.shape == v2.observation_space.shape


@pytest.mark.parametrize(
    "domain_name, seed",
    [
        ("dolphine3", 0),
        ("random1_1", 1),
        ("test1", 42),
    ],
)
def test_env_equivalence_multiple_domains_and_seeds(domain_name: str, seed: int) -> None:
    """Stronger equivalence: several domains and seeds should match step-by-step."""

    legacy, v2 = _make_envs(domain_name)

    random.seed(seed)
    np.random.seed(seed)

    obs_legacy = legacy.reset()
    obs_v2, _info_v2 = v2.reset()

    assert isinstance(obs_legacy, np.ndarray)
    assert isinstance(obs_v2, np.ndarray)
    assert obs_legacy.shape == obs_v2.shape

    for _ in range(30):
        action = v2.action_space.sample()
        o1, r1, d1, info1 = legacy.step(action)
        o2, r2, term2, trunc2, info2 = v2.step(action)
        d2 = bool(term2 or trunc2)

        assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
        assert o1.shape == o2.shape
        assert math.isfinite(float(r1)) and math.isfinite(float(r2))
        assert math.isclose(float(r1), float(r2), rel_tol=1e-9, abs_tol=1e-9)
        assert isinstance(d1, (bool, np.bool_)) and isinstance(d2, (bool, np.bool_))
        assert d1 == d2
        assert "is_complete" in info1 and "is_complete" in info2
        assert bool(info1["is_complete"]) == bool(info2["is_complete"])

        if d1:
            break
