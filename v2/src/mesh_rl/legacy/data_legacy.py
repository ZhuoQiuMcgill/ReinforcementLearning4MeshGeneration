"""Legacy-style geometric data utilities used by the v2 mesh_rl package.

This module is a v2-local reimplementation of the small helper functions
from general.data (matrix_ops, transformation, detransformation, etc.).
It exists so that v2 does not import or modify any code in the original
project while keeping the numerical behaviour identical.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def get_patterns(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pattern data from the legacy text format.

    This mirrors general.data.get_patterns so tests and any legacy-style
    utilities can continue to work in a self-contained way under v2.
    """

    pattern_inputs = []
    pattern_outputs = []
    pattern_types = []
    with open(filename, "r+") as fr:
        for line in fr:
            if not line.startswith("%"):
                line_dat = [float(r) for r in line.split()]
                pattern_inputs.append(line_dat[2:12])
                pattern_types.append([line_dat[14]])
                pattern_outputs.append(line_dat[15:17])

    return (
        np.asarray(pattern_inputs),
        np.asarray(pattern_types),
        np.asarray(pattern_outputs),
    )


def matrix_ops(arra: Iterable[float]) -> np.matrix:
    """Convert a flat sequence [x0, y0, x1, y1, ...] into a 2D matrix.

    Behaviour is identical to general.data.matrix_ops.
    """

    arra = np.asarray(arra, dtype=float)
    matrix = np.split(arra, len(arra) / 2)
    matrix = np.asmatrix(matrix)
    return matrix


def transformation(matrix: np.ndarray, dist: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Apply the legacy normalization + rotation to a set of points.

    Parameters match general.data.transformation.
    """

    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix - p0

    # dist is the reference distance (typically ||p0 - p1||)
    matrix = np.divide(matrix, dist)

    theta = math.atan2((p1 - p0)[1], (p1 - p0)[0])

    rotation_matrix = np.asmatrix(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )
    matrix = np.matmul(rotation_matrix, matrix.T).T

    return np.asarray(matrix).reshape(-1)


def data_transformation(
    data: np.ndarray,
    ind_x_1: int,
    ind_y_1: int,
    ind_x0: int,
    ind_y0: int,
    ind_x1: int,
    ind_y1: int,
) -> np.ndarray:
    """Vectorized wrapper around matrix_ops + transformation.

    This is a direct port of general.data.data_transformation.
    """

    data = np.asarray(data, dtype=float)
    line_len = len(data[0])  # kept for parity with the legacy code
    _ = line_len

    data_transf = []
    for line in data:
        mat = matrix_ops(line)
        data_transf.append(
            transformation(
                mat,
                np.asarray([line[ind_x_1], line[ind_y_1]]),
                np.asarray([line[ind_x0], line[ind_y0]]),
                np.asarray([line[ind_x1], line[ind_y1]]),
            )
        )

    return np.asarray(data_transf)


def detransformation(point: np.ndarray, dist: float, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Inverse of transformation.

    This mirrors general.data.detransformation but uses math instead of
    np.math for better compatibility with modern NumPy.
    """

    point = np.asarray(point, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    theta = 2 * math.pi - math.atan2((p1 - p0)[1], (p1 - p0)[0])
    original_point = np.empty(2)

    # remove rotation
    original_point[0] = math.cos(theta) * point[0] + math.sin(theta) * point[1]
    original_point[1] = -math.sin(theta) * point[0] + math.cos(theta) * point[1]

    # remove scaling
    original_point *= dist

    # remove translation
    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point
