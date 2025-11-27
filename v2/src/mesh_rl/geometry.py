"""Geometry primitives and helpers for the v2 mesh RL project.

This module exposes the core geometry and mesh primitives used by the
v2 pipeline, plus polygon loading from JSON domain files and a minimal
``MeshFrame`` used by the rendering code in the environments.

The heavy geometric logic itself lives in :mod:`mesh_rl.components_core`
and :mod:`mesh_rl.mesh_core`, which are v2-local copies of the original
implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt

from mesh_rl.components_core import (
    Point2D,
    Vertex,
    Segment,
    Boundary2D,
    Mesh,
    PointEnvironment,
)
from mesh_rl.mesh_core import MeshGeneration


# ---- polygon loading ----------------------------------------------------


def read_polygon(filename: str | Path) -> Boundary2D:
    """Read a polygon from a JSON list of [x, y] pairs.

    This is a v2-local equivalent of ``general.polygon.read_polygon``.
    The JSON file is expected to contain a single line with a list of
    ``[x, y]`` coordinates in the original pixel scale; we rescale by
    ``1 / 100`` to obtain coordinates in the working mesh space.
    """

    path = Path(filename)
    with path.open("r", encoding="utf-8") as fr:
        vertices = json.loads(fr.readline())
    points = [Vertex(p[0] / 100.0, p[1] / 100.0) for p in vertices]
    # Connect consecutive vertices with segments
    for i in range(len(points)):
        seg = Segment(points[i - 1], points[i])
        points[i - 1].assign_segment(seg)
        points[i].assign_segment(seg)
    return Boundary2D(points)


# ---- MeshFrame used for rendering --------------------------------------


class MeshFrame:
    """Minimal wrapper used by :class:`BoudaryEnv` for line drawing.

    The original implementation relied on pygame; here we expose a small
    abstraction over matplotlib-compatible coordinates so that v2 can run
    without additional GUI dependencies.
    """

    def __init__(self, window_size: Tuple[int, int]) -> None:
        self.window_size = window_size

    def draw_line(self, p1: Tuple[float, float], p2: Tuple[float, float], color: Tuple[float, float, float]) -> None:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)

    def render(self) -> None:
        plt.gca().set_aspect("equal", adjustable="box")
        plt.pause(0.001)

    def close(self) -> None:
        plt.close()
