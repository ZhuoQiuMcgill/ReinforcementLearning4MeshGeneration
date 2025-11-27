"""Algorithm registry for the v2 mesh RL pipeline.

This module provides a mapping from simple algorithm names ("sac",
"ppo", etc.) to Stable-Baselines3 classes and default hyperparameters.
The goal is to centralise RL-library-specific details so that the
higher-level training code can remain declarative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import torch as th
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from ..config import RLConfig
from ..config import resolve_device


@dataclass
class AlgoSpec:
    """Specification for a single SB3 algorithm variant."""

    cls: Type[BaseAlgorithm]
    policy: str
    default_kwargs: Dict[str, Any]


_ALGOS: Dict[str, AlgoSpec] = {
    # A2C / DDPG currently rely mostly on SB3 defaults in the legacy
    # script; we keep them minimal here and may refine later.
    "a2c": AlgoSpec(
        cls=A2C,
        policy="MlpPolicy",
        default_kwargs={},
    ),
    "ddpg": AlgoSpec(
        cls=DDPG,
        policy="MlpPolicy",
        default_kwargs={},
    ),
    # PPO: mirror the legacy policy architecture
    "ppo": AlgoSpec(
        cls=PPO,
        policy="MlpPolicy",
        default_kwargs={
            "policy_kwargs": dict(
                activation_fn=th.nn.ReLU,
                net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            ),
        },
    ),
    # SAC: mirror legacy net_arch and core hyperparameters
    "sac": AlgoSpec(
        cls=SAC,
        policy="MlpPolicy",
        default_kwargs={
            "policy_kwargs": dict(
                activation_fn=th.nn.ReLU,
                net_arch=[128, 128, 128],
            ),
            "learning_rate": 3e-4,
            "learning_starts": 10_000,
            "batch_size": 100,
        },
    ),
    # TD3: mirror legacy net_arch and learning parameters
    "td3": AlgoSpec(
        cls=TD3,
        policy="MlpPolicy",
        default_kwargs={
            "policy_kwargs": dict(
                activation_fn=th.nn.ReLU,
                net_arch=[256, 256],
            ),
            "learning_rate": 3e-4,
            "learning_starts": 10_000,
        },
    ),
}


def get_algo_spec(algo: str) -> AlgoSpec:
    key = algo.lower()
    if key not in _ALGOS:
        raise KeyError(f"Unsupported algorithm: {algo!r}")
    return _ALGOS[key]


def build_model(algo: str, env, cfg: RLConfig, **extra_kwargs: Any) -> BaseAlgorithm:
    """Instantiate an SB3 model for the given algorithm and env.

    The defaults in :data:`_ALGOS` mirror the hyperparameters that were
    hard-coded in the legacy ``RL_Mesh.py`` script (policy network
    architectures, learning rate, batch size, etc.). Callers can
    override *any* of these via ``cfg.algo_kwargs`` or ``extra_kwargs``.
    ``extra_kwargs`` takes precedence over ``cfg.algo_kwargs`` which in
    turn overrides the legacy defaults.
    """

    spec = get_algo_spec(algo)
    device = resolve_device(cfg.device)
    # Start from legacy defaults, then allow RLConfig.algo_kwargs and
    # explicit call-site kwargs to override them.
    kwargs: Dict[str, Any] = dict(spec.default_kwargs)
    kwargs.update(cfg.algo_kwargs)
    kwargs.update(extra_kwargs)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("tensorboard_log", None)
    kwargs.setdefault("device", device)

    return spec.cls(spec.policy, env, **kwargs)
