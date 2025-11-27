"""v2 implementation of the BoudaryEnv environment.

All dependencies (geometry, polygon IO, rendering) are resolved inside
v2 so that ``mesh_rl`` is completely independent from the legacy
repository modules.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from mesh_rl.data_core import matrix_ops, transformation, detransformation
from mesh_rl.mesh_core import MeshGeneration
from mesh_rl.components_core import (
    Vertex,
    Segment,
    Boundary2D,
    Mesh,
    PointEnvironment,
)
from mesh_rl.geometry import read_polygon, MeshFrame


class BoudaryEnv(MeshGeneration, gym.Env):
    """Gym-style environment for quadrilateral mesh generation.

    This is the v2 copy of the original ``rl.boundary_env.BoudaryEnv``.
    All mathematical logic (state, action, reward, termination) must
    remain semantically identical to the legacy implementation.
    Only readability-oriented refactors are allowed here.
    """

    TYPE_THRESHOLD = 0.3

    @classmethod
    def from_domain_file(
        cls,
        filename: str,
        *,
        experiment_version: Optional[str] = None,
        env_name: Optional[int] = None,
    ) -> "BoudaryEnv":
        """Construct an environment from a domain JSON file path."""

        boundary_obj = read_polygon(filename)
        return cls(boundary_obj, experiment_version=experiment_version, env_name=env_name)

    def __init__(self, boundary: Boundary2D, experiment_version: Optional[str] = None, env_name: Optional[int] = None) -> None:
        """Initialize a new BoudaryEnv instance.

        Parameters
        ----------
        boundary:
            Geometric boundary describing the initial domain to be meshed.
        experiment_version:
            Optional experiment tag used in logging paths.
        env_name:
            Optional numeric id used to distinguish parallel environments.
        """
        super(BoudaryEnv, self).__init__(boundary)
        self.original_boundary = self.boundary.deep_copy()
        self.original_area = self.boundary.poly_area()
        self.current_area = self.original_area
        self.max_radius = 2 # change from 3 to 2

        # Use float32 for Box bounds to avoid Gymnasium precision warnings
        # when ``dtype`` is float32.
        act_low = np.array([-1.0, -1.5, 0.0], dtype=np.float32)
        act_high = np.array([1.0, 1.5, 1.5], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -2, 0]), np.array([1, 2, 2]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -1, 0]), np.array([1, 1, 1]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -3, 0]), np.array([1, 3, 3]), dtype=np.float32)

        self.neighbor_num = 6 # from 4 to 6
        self.radius_num = 3
        self.POOL_SIZE = 10
        self.radius = 4

        # Observation space bounds in float32 to silence precision warnings.
        obs_shape = (2 * (self.neighbor_num + self.radius_num),)
        obs_low = np.full(obs_shape, -999.0, dtype=np.float32)
        obs_high = np.full(obs_shape, 999.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # self.observation_space = spaces.Box(low=-999, high=999,
        #                                     shape=(self.POOL_SIZE, 2 * (self.neighbor_num + self.radius_num)), dtype=np.float32)
        self.current_point_environment = None
        self.test_point_environments = []
        self.no_state_change = 0
        self.smoothed = False
        self.MIN_TRANSITION_QUALITY = 0.2
        self.not_valid_points = []
        self.last_not_valid_points = []

        self.reward_method = 10 # from 2 to 0
        self.last_index = 0
        self.target_angle = 0
        self.rewarding = []
        self.estimated_area_range = None
        self.window_size = (500, 500)
        self.current_state = None

        # loggging
        self.experiment_version = experiment_version if experiment_version else 'test'
        self.env_name = env_name if env_name is not None else 1
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seeds for Python and NumPy.

        This is added for compatibility with Gym/Gymnasium wrappers and
        Stable-Baselines3. Behaviourally, the legacy environment relied
        on global RNG state; this method provides an explicit hook
        without changing core logic.
        """

        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        static: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Reset environment state to the original boundary.

        This signature is compatible with Gymnasium/SB3, but we keep the
        original behaviour: seeding updates the global RNGs and the
        method returns only the observation array.

        Parameters
        ----------
        seed:
            Optional seed for Python and NumPy RNGs.
        static:
            If True, use a static point-environment when selecting the
            next state.
        options:
            Unused Gymnasium options dict (accepted for API compat).

        Returns
        -------
        numpy.ndarray
            The initial observation vector for the new episode.
        """
        if seed is not None:
            self.seed(seed)

        self.viewer = None
        self.boundary = self.original_boundary.deep_copy()
        self.updated_boundary = self.boundary.copy()
        self.original_vertices = [v for v in self.boundary.vertices]
        self.generated_meshes = []
        self.not_valid_points = []
        self.rewarding = []
        self.current_area = self.original_area
        self.current_point_environment = None
        self.test_point_environments = []
        self.candidate_vertices = None
        self.test_candidate_vertices = []
        self.failed_num = 0
        state = self.find_next_state(static=static)
        self.current_state = state
        self.estimated_area_range = self.estimate_area_range()
        info: Dict[str, Any] = {}
        return state, info

    def get_middle_points(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the two middle points used as a local reference frame."""

        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def transformation(self) -> np.ndarray:
        """Compute the normalized + rotated representation of current state."""

        arra = self.current_point_environment.state
        p0, p1 = self.get_middle_points(arra)

        mat = matrix_ops(arra)
        return transformation(mat, self.current_point_environment.base_length, p0, p1)

    def detransformation(self, point: Sequence[float], is_move: bool = False) -> Vertex:
        # v1, v2 = self.get_middle_points(self.current_point_environment.state)
        v1, v2 = self.get_middle_points(self.current_point_environment.points_as_array(self.current_point_environment.neighbors))
        # _point = [self.current_point_environment.base_length * self.radius * point[1] * math.cos(point[0]),
        #           self.current_point_environment.base_length * self.radius * point[1] * math.sin(point[0])]
        # de_point = detransformation(_point, self.current_point_environment.base_length, v1, v2)

        de_point = detransformation(point, self.current_point_environment.base_length if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def find_ideal_mesh(self, v_l: Vertex, v: Vertex, v_r: Vertex) -> Mesh:
        """Return the ideal quadrilateral mesh given three boundary vertices."""

        can_v1, can_v2 = v_l.get_perpendicular_vertex(v_r)
        can_v = can_v1 if v.distance_to(can_v1) >= v.distance_to(can_v2) else can_v2
        return Mesh([v_l, v, v_r, can_v])

    def _build_mesh_for_step(
        self,
        *,
        rule_type: float,
        index: int,
        new_point: Vertex,
    ) -> Tuple[Optional[Mesh], int, float]:
        """Construct the candidate mesh and rule id for a given action.

        This is a direct structural refactor of the rule-selection branch in
        :meth:`step`. It returns the mesh (if any), the discrete rule id and
        any immediate reward delta incurred by invalid actions.
        """

        reward_delta = 0.0
        mesh: Optional[Mesh]

        if rule_type <= -0.5:  # self.TYPE_THRESHOLD in legacy code
            mesh = Mesh(
                [
                    self.updated_boundary.vertices[index - 1],
                    self.updated_boundary.vertices[index],
                    self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                ]
            )
            rule = -1
        elif rule_type >= 0.5:  # 1 - self.TYPE_THRESHOLD in legacy code
            mesh = Mesh(
                [
                    self.updated_boundary.vertices[index - 2],
                    self.updated_boundary.vertices[index - 1],
                    self.updated_boundary.vertices[index],
                    self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                ]
            )
            rule = 1
        else:
            # reward -= 0.1 * math.fabs(rule_type)
            if self.is_point_inside_area(new_point):
                existing_new_point = self.find_same_point(new_point)
                if existing_new_point:
                    mesh = Mesh(
                        [
                            self.updated_boundary.vertices[index - 1],
                            self.updated_boundary.vertices[index],
                            self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                            self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                        ]
                    )
                else:
                    mesh = Mesh(
                        [
                            new_point,
                            self.updated_boundary.vertices[index - 1],
                            self.updated_boundary.vertices[index],
                            self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        ]
                    )
            else:
                reward_delta += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1
                mesh = None
            rule = 0

        return mesh, rule, reward_delta

    def _apply_mesh_if_valid(
        self,
        *,
        mesh: Mesh,
        reference_point: Vertex,
        rule: int,
        reward: float,
        done: bool,
    ) -> Tuple[float, bool, bool]:
        """Apply a candidate mesh if it passes all geometric checks.

        This is a straight refactor of the inner part of :meth:`step` that
        handled mesh validation, boundary updates, reward computation and
        termination conditions. The numerical behaviour is intended to be
        identical to the legacy implementation.

        Parameters
        ----------
        mesh:
            Candidate quadrilateral element around the current reference
            point.
        reference_point:
            Current reference vertex on the boundary.
        rule:
            Discrete rule type used as key in ``history_info`` (-1, 0, 1).
        reward:
            Current cumulative reward for this step.
        done:
            Whether the episode is already marked as finished.

        Returns
        -------
        reward:
            Updated reward after applying (or rejecting) the mesh.
        done:
            Updated done flag.
        failed:
            Whether the action is considered to have failed.
        """

        failed = True

        if self.validate_mesh(mesh, quality_method=0) and not self.check_intersection_with_boundary(
            mesh, reference_point
        ):
            mesh.connect_vertices()
            self.generated_meshes.append(mesh)

            # Update boundary and reference points.
            self.update_boundary(reference_point, mesh)
            mesh_area = mesh.compute_area()[0]
            self.current_area -= mesh_area

            quality = self.get_quality(mesh, 2)
            speed_penalty = self.get_speed_penalty(mesh_area, reference_point)
            reward += quality + speed_penalty

            self.history_info[rule].append(reward)

            failed = False
            if len(self.updated_boundary.vertices) <= 5:
                reward += 10
                done = True
                if len(self.updated_boundary.vertices) == 4:
                    final_mesh = Mesh(self.updated_boundary.vertices)
                    final_mesh.connect_vertices()
                    self.generated_meshes.append(final_mesh)
            else:
                done = False

        else:
            # Penalise invalid meshes in the same way as the legacy code.
            reward += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1

        return reward, done, failed

    def _finalize_step(
        self,
        *,
        reward: float,
        done: bool,
        failed: bool,
    ) -> Tuple[Optional[np.ndarray], np.float64, bool, Dict[str, Any]]:
        """Compute next_state, update failure counters and build info dict.

        This is a direct refactor of the tail of :meth:`step`, which was
        responsible for selecting the next reference point, updating
        ``failed_num`` and determining whether the episode is complete.
        """

        is_complete = True
        next_state = self.find_next_state(self.not_valid_points, last_failed=failed)

        if not failed:
            self.failed_num = 0
        else:
            self.failed_num += 1
            if self.failed_num >= 100:  # self.action_space.shape[0] * 40
                done = True
                is_complete = False

        return next_state, np.float64(reward), done, {"is_complete": is_complete}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, bool, Dict[str, Any]]:
        """Apply one environment step given an action.

        This method mutates the internal boundary/mesh state and returns
        the next observation, reward, terminated flag, truncated flag and
        info dict, following the Gymnasium 5-tuple API. Internally we
        keep the exact same ``done``/``is_complete`` semantics as the
        legacy implementation and map them to ``terminated``/``truncated``
        as follows:

        * ``terminated = done and is_complete``
        * ``truncated = done and not is_complete``
        """
        done = False
        failed = True
        reward = 0

        # rp_index = int(action[0]*10)
        # rule_type = action[1]

        skip = 0.9
        rule_type = action[0]
        rule = None
        new_point = self.action_2_point(action[1:])
        # print(skip)
        # if rp_index >= len(self.test_point_environments) - 1:
        if skip < 0.5:
            reward += -1
        else:
            # self.current_point_environment = self.test_point_environments[rp_index]

            # new_point = self.action_2_point(action[2:])

            reference_point = self.current_point_environment.reference_point
            index = self.updated_boundary.vertices.index(reference_point)

            # print(action, rule_type, new_point)
            # new_point.show('r.')
            # self.boundary.show()

            if len(self.updated_boundary.vertices) <= 5:
                reward = 10
                done = True
            else:

                mesh, rule, reward_delta = self._build_mesh_for_step(
                    rule_type=rule_type,
                    index=index,
                    new_point=new_point,
                )
                reward += reward_delta

                if mesh is not None:
                    reward, done, failed = self._apply_mesh_if_valid(
                        mesh=mesh,
                        reference_point=reference_point,
                        rule=rule,
                        reward=reward,
                        done=done,
                    )

        next_state, reward_arr, done, info = self._finalize_step(
            reward=reward,
            done=done,
            failed=failed,
        )
        is_complete = bool(info.get("is_complete", False))
        terminated = bool(done and is_complete)
        truncated = bool(done and not is_complete)
        return next_state, reward_arr, terminated, truncated, info

    def move(self, new_point, type, lr_1=None, lr_2=None):
        """Deterministically apply a geometric move, mirroring legacy behaviour.

        This is primarily used by data-generation utilities, not by the
        RL training loop. The implementation is kept structurally close to
        the original code, but uses the same internal helpers for mesh
        construction and state updates where appropriate.
        """
        x, y = (
            self.current_point_environment.base_length
            * self.radius
            * new_point[0]
            * math.cos(new_point[1]),
            self.current_point_environment.base_length
            * self.radius
            * new_point[0]
            * math.sin(new_point[1]),
        )
        new_point = self.detransformation([round(x, 6), round(y, 6)], is_move=True)

        done = False
        next_state = None
        not_valid_element = True

        reference_point = self.current_point_environment.reference_point

        if len(self.updated_boundary.vertices) <= 5:
            reward = 10
            done = True

        else:
            index = self.updated_boundary.vertices.index(reference_point)

            # The move() function uses TYPE_THRESHOLD instead of the [-0.5, 0.5]
            # splits from step(), so we keep its branching logic separate.
            mesh: Optional[Mesh] = None
            if type <= self.TYPE_THRESHOLD:
                mesh = Mesh(
                    [
                        self.updated_boundary.vertices[index - 1],
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                    ]
                )
            elif type >= 1 - self.TYPE_THRESHOLD:
                mesh = Mesh(
                    [
                        self.updated_boundary.vertices[index - 2],
                        self.updated_boundary.vertices[index - 1],
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                    ]
                )
            else:
                if self.is_point_inside_area(new_point):
                    mesh = Mesh(
                        [
                            new_point,
                            self.updated_boundary.vertices[index - 1],
                            self.updated_boundary.vertices[index],
                            self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                        ]
                    )

            if mesh is not None and self.validate_mesh(mesh, quality_method=0) and not self.check_intersection_with_boundary(
                mesh, reference_point
            ):
                mesh.connect_vertices()
                not_valid_element = False
                self.generated_meshes.append(mesh)

                self.update_boundary(reference_point, mesh)

                next_state = self.find_next_state(self.not_valid_points, static=True)

                if len(self.updated_boundary.vertices) <= 5:
                    done = True
                    if len(self.updated_boundary.vertices) == 4:
                        mesh = Mesh(self.updated_boundary.vertices)
                        self.generated_meshes.append(mesh)

            # Legacy "old handling" block for not_valid_points management.
            if not_valid_element:
                if reference_point not in self.not_valid_points:
                    self.not_valid_points.append(reference_point)
                next_state = self.find_next_state(self.not_valid_points, static=True)
            else:
                self.not_valid_points = []

            # The commented-out "new handling" logic is intentionally
            # preserved but kept disabled, as in the original source.

            if len(self.updated_boundary.vertices) > 4:
                is_complete = False
                if next_state is None:
                    if lr_1 and lr_2:
                        self.smooth_pave(
                            self.boundary.vertices,
                            self.updated_boundary.vertices,
                            lr_1,
                            lr_2,
                            iteration=400,
                        )
                    else:
                        self.smooth_pave(
                            self.boundary.vertices,
                            self.updated_boundary.vertices,
                            iteration=400,
                        )

                    if len(self.last_not_valid_points) > 0 and len(self.not_valid_points) > 0:
                        if (
                            self.last_not_valid_points[0] == self.not_valid_points[0]
                            and self.last_not_valid_points[-1] == self.not_valid_points[-1]
                            and len(self.not_valid_points) == len(self.last_not_valid_points)
                        ):
                            done = True

                    self.last_not_valid_points = self.not_valid_points
                    self.not_valid_points = []

                    next_state = self.find_next_state(not_valid_points=self.not_valid_points, static=True)
                    if next_state is None:
                        done = True

            else:
                is_complete = True

        return next_state, 0, done, {"is_complete": is_complete}

    def get_speed_penalty(self, mesh_area: float, reference_p: Vertex) -> float:
        """Compute the speed-regularisation term based on element area."""

        # _ind = mesh.vertices.index(reference_p)
        # dist = (reference_p.distance_to(mesh.vertices[_ind - 1]) +
        #         reference_p.distance_to(mesh.vertices[_ind - 3])) / 2

        # min_area = max(self.estimated_area_range[0] ** 2, 0.5 * dist ** 2)
        min_area = self.estimated_area_range[0] ** 2 #* 0.5 #*1.5
        critical_area = self.estimated_area_range[1] ** 2 #* 0.5# *1.5

        if min_area <= mesh_area < critical_area:
            speed_penalty = ((mesh_area - critical_area) / (critical_area - min_area))
        elif mesh_area < min_area:
            speed_penalty = -1
        else:
            speed_penalty = 0
        return speed_penalty

    def get_reward(self, mesh: Mesh, method: int = 0) -> float:
        """Legacy reward combinator used in some experimental modes."""

        reward = None

        if method == 0:
            mesh_quality = mesh.get_quality()
            reward = mesh_quality
        elif method == 1:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)
            reward = mesh_quality * transition_quality
        elif method == 2:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = mesh_quality * transition_quality + forward_quality
        elif method == 3:
            mesh_quality = mesh.get_quality()
            forward_quality = 5 / len(self.updated_boundary.vertices)
            reward = forward_quality + mesh_quality

        elif method == 4:
            forward_quality = 5 / len(self.updated_boundary.vertices)
            reward = forward_quality

        elif method == 5:
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = transition_quality + forward_quality
        elif method == 9:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = mesh_quality * transition_quality * forward_quality
        elif method == 10:

            reward = self.get_quality(mesh, index=1)

        return reward

    def conduct_smooth(self, lr_1: float = 0.999, lr_2: float = 0.999, iteration: int = 200) -> None:
        """Run a local smoothing pass around the current reference point."""

        index = self.updated_boundary.vertices.index(self.current_point_environment.reference_point)
        self.smooth(self.boundary.vertices, lr_1, lr_2, iteration)
        # self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices,
        #                  lr_1, lr_2, iteration=400)
        self.current_point_environment = PointEnvironment(self.updated_boundary.vertices[index], self.updated_boundary)

    def find_next_state(
        self,
        not_valid_points: Optional[List[Vertex]] = None,
        last_failed: bool = False,
        static: bool = False,
    ) -> Optional[np.ndarray]:
        """Select the next reference point and build the corresponding state."""
        # self.boundary.show()
        r_p = None
        # if len(self.next_references):
        #     max_angle = 120
        #     while len(self.next_references):
        #         v, angle = self.next_references[0]
        #         if not_valid_points:
        #             if v in not_valid_points:
        #                 self.next_references.pop(0)
        #                 continue
        #         if angle <= max_angle:
        #             r_p = v
        #             break
        #         else:
        #             self.next_references.pop(0)

        # if len(self.test_candidate_vertices) == 0:
        #     self.find_reference_candidates(target_angle=0)
        #
        # if not last_failed:
        #     rp_length = len(self.test_candidate_vertices) if len(self.test_candidate_vertices) < self.POOL_SIZE \
        #         else self.POOL_SIZE
        #
        #     self.test_point_environments = [
        #         PointEnvironment(reference_point=self.test_candidate_vertices[i][0], boundary=self.updated_boundary,
        #                            neighbor_num=self.neighbor_num, radius_num=self.radius_num,
        #                            average_edge_length=self.average_edge_length,
        #                            area_ratio=self.current_area / self.original_area) for i in range(rp_length)]
        #     state = [p.state for p in self.test_point_environments]
        #     state.extend([[0] * 2 * (self.neighbor_num + self.radius_num) for i in range(self.POOL_SIZE - rp_length)])
        #     self.current_state = state

        # return np.array(self.current_state).astype(np.float32)

        if r_p is None:
            r_p = self.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            # print(r_p)
            p_e = PointEnvironment(reference_point=r_p, boundary=self.updated_boundary,
                                   neighbor_num=self.neighbor_num, radius_num=self.radius_num,
                                   average_edge_length=self.average_edge_length,
                                   area_ratio=self.current_area / self.original_area,
                                   radius=self.radius, static=static)
            self.current_point_environment = p_e

            # state = ([round(elem, 4) for elem in self.transformation()])
            state = p_e.state
            # print(state)

            # self.boundary.show(show=False)
            # for p in self.current_point_environment.state_vertices:
            #     if p is not None:
            #         p.show('r.')
            # self.current_point_environment.state_vertices[0].show('k.')
            # plt.show()

            # if len(p_e.radius_neighbors):
            #     [state.append(round(elem, 4)) for elem in self.transformation(p_e.state, self.points_as_array(p_e.radius_neighbors))]
            # state.append(p_e.available_radius)
            # add area ratio
            # state.append(self.current_area / self.original_area)
            return np.array(state).astype(np.float32)       # return self.points_as_array(neighbors)

        else:
            # self.last_point_environment = None
            return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def render(self, mode: str = 'human') -> None:
        print(f'Generated elements: {len(self.generated_meshes)}')
        if self.viewer is None:
            # self.viewer = rendering.Viewer(500, 500)
            # self.viewer.set_bounds(-1, 12, -6.5, 6.5)
            # self.times = 1
            self.viewer = MeshFrame(self.window_size)
            min_x, max_x = min([v.x for v in self.original_vertices]), max([v.x for v in self.original_vertices])
            min_y, max_y = min([v.y for v in self.original_vertices]), max([v.y for v in self.original_vertices])
            self.times = int(min(self.window_size[0] / (max_x-min_x), self.window_size[1] / (max_y - min_y)))
            self.min_x, self.min_y = min_x, min_y

        all_segts = self.boundary.all_segments()
        for line in all_segts:
            self.viewer.draw_line(((line.point1.x - self.min_x) * self.times, (line.point1.y - self.min_y) * self.times),
                                  ((line.point2.x - self.min_x) * self.times, (line.point2.y - self.min_y) * self.times),
                                  color=(0, 0, 1)) #color=(0, 0, 1)
        time.sleep(.0001)
        self.viewer.render()
        # self.boundary.show()

    def find_same_point(self, point: Vertex) -> Optional[Vertex]:
        for p in self.updated_boundary.vertices:
            if p.distance_to(point) < 0.001:
                return p

    def state_2_points(self, state: Sequence[float]) -> List[Vertex]:
        '''

        :param state: A list of values
        :return: a list of Vertices
        '''
        if len(state) % 2 != 0:
            raise ValueError('The lenght of state is not correct.')

        return [Vertex(state[i * 2], state[i * 2 + 1]) for i in range(int(len(state)/2))]


    def action_2_point(self, action: Any) -> Vertex:
        if isinstance(action, np.ndarray):
            x, y = action[0], action[1]
            # x, y = 1.5 * 1.4142 * action[0] * math.cos(math.pi * action[1]), \
            #        1.5 * 1.4142 * action[0] * math.sin(math.pi * action[1])
        else:
            n_action = [action / (self.max_radius * 10), (action % (self.max_radius * 10)) / 10]
            x = round(math.cos(math.radians(n_action[0])), 1) * n_action[1]
            y = round(math.sin(math.radians(n_action[0])), 1) * n_action[1]
        return self.detransformation([round(x, 4), round(y, 4)])

    def random_action(self, state: Sequence[float]) -> int:
        v_p = [state[2], state[3]]
        theta = int(math.degrees(math.asin(v_p[1]/math.sqrt(v_p[1] ** 2 + v_p[0] ** 2))))
        if theta < 0:
            theta += 360
        while True:
            # action = random.randint(0, self.action_space - 1)
            random_theta = random.randint(0, theta)
            random_radius = random.randint(0, self.current_point_environment.available_radius * 10)
            action = random_theta * self.max_radius * 10 + random_radius
            v = self.action_2_point(action)
            if self.is_point_inside_area(v):
                return action

    def is_action_valid(self, action: Any, state: Sequence[float]) -> bool:
        v = self.action_2_point(action)
        if self.is_point_inside_area(v):
            return True
        else:
            return False

    def write_2_file(self, filename: str) -> None:
        nodes = {}
        for i, v in enumerate(self.boundary.vertices):
            nodes[i] = {"coordinates": [v.x, v.y],
                        "connected": []}

        for i, v in enumerate(self.boundary.vertices):
            for seg in v.segments:
                if seg.point1 is v:
                    partner = seg.point2
                else:
                    partner = seg.point1
                if partner not in nodes[i]['connected']:
                    nodes[i]['connected'].append(self.boundary.vertices.index(partner))

        elements = {}
        for i, ele in enumerate(self.generated_meshes):
            elements[i] = [self.boundary.vertices.index(v) for v in ele.vertices]

        with open(filename, 'w') as fw:
            json.dump({'nodes': nodes, 'elements': elements}, fw)
            fw.close()

    @staticmethod
    def read_2_object(filename: str) -> "BoudaryEnv":
        with open(filename, 'r') as fr:
            data = json.load(fr)
            fr.close()

        vertices = {}
        for x, y in data['nodes'].items():
            vertices[x] = Vertex(y['coordinates'][0], y['coordinates'][1])

        for x, y in data['nodes'].items():
            for partner in y['connected']:
                sg = Segment(vertices[x], vertices[str(partner)])
                if vertices[x].segments is not None:
                    for s in vertices[x].segments:
                        if vertices[x] in [s.point1, s.point2] and vertices[str(partner)] in [s.point1, s.point2]:
                            break
                    else:
                        vertices[x].assign_segment(sg)
                        vertices[str(partner)].assign_segment(sg)
                else:
                    vertices[x].assign_segment(sg)
                    vertices[str(partner)].assign_segment(sg)

        env = BoudaryEnv(Boundary2D(list(vertices.values())))

        elements = []

        for i, e in data["elements"].items():
            _vertices = [vertices[str(id)] for id in e]
            element = Mesh(_vertices)
            elements.append(element)
        plt.gca().set_aspect('equal', adjustable='box')
        env.generated_meshes = elements

        segts = env.boundary.all_segments()
        for segt in segts:
            if segt.point1 not in env.boundary.vertices or segt.point2 not in env.boundary.vertices:
                continue
            segt.show(style='k-')

        circle2 = plt.Circle((10.09, -0.13), 3.3, color='black', linestyle='--', fill=False)
        # ax.add_artist(circle2)
        plt.gcf().gca().add_artist(circle2)
        # env.plot_points(elements[15].vertices)
        # p_l_r = [elements[15].vertices[0], elements[15].vertices[3], elements[15].vertices[2], elements[15].vertices[1],
        #          vertices['15']]
        # p_theta = [vertices['20'], vertices['44'], vertices['46']]
        # p_l_r = [elements[15].vertices[0], elements[15].vertices[1], elements[15].vertices[2], elements[15].vertices[3],
        #          vertices['34']]
        # p_theta = [vertices['20'], vertices['44'], vertices['46']]
        p_l_r = [ vertices['15'], elements[15].vertices[1], elements[15].vertices[2], elements[15].vertices[3],
                 vertices['34']]
        p_theta = [vertices['20'], vertices['44'], vertices['46']]

        env.plot_points(p_l_r, 'r-o')
        env.plot_points(p_theta)
        env.plot_points([elements[15].vertices[0]], 'ko')

        plt.show()

        #env.plot_points([vertices['15'], vertices['20'],  vertices['44'], vertices['46']])
        #env.plot_points([vertices['15']])

        #env.plot_points([vertices['34'], vertices['20'],  vertices['44'], vertices['46']])
        #env.plot_points([vertices[34], vertices[20],  vertices[44], vertices[46]])

        #
        env.save_meshes("D:\\meshingData\\test_1.png", env.generated_meshes, quality=True,
                        indexing=True,
                        type=1, dpi=300)
        print()
        env.extract_samples(elements)

        samples, output_types, outputs = env.extract_samples(elements)
        env.save_samples(f'D:\\meshingData\\A2C\\domain\\ebrd_1.json',
                         {'samples': samples, 'output_types': output_types, 'outputs': outputs})


    def plot_sample(self, state: Sequence[float], action: Sequence[float]) -> None:
        L = len(state)
        N = int((L - 6) / 2)
        left = [-i - 2 for i in reversed(range(0, N, 2))]
        right = [i for i in range(0, N, 2)]
        all = left + right

        medium = [N + i for i in range(0, 6, 2)]

        n_x = [state[j] * math.cos(state[j + 1]) for j in all]
        n_x.insert(int(N/2), 0)
        r_x = [state[j] * math.cos(state[j + 1]) for j in medium]
        n_y = [state[j] * math.sin(state[j + 1]) for j in all]
        n_y.insert(int(N/2), 0)
        r_y = [state[j] * math.sin(state[j + 1]) for j in medium]
        plt.plot(n_x, n_y, 'k.-')
        plt.plot(r_x, r_y, 'y.-')
        target_x, target_y = action[1] * math.cos(action[2]), action[1] * math.sin(action[2])
        plt.plot(target_x, target_y, 'r.')
        plt.title(f'type: {action[0]}')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def save_history_info(self, filename: str) -> None:
        with open(filename, 'w') as fw:
            json.dump(self.history_info, fw)
            fw.close()


# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\plots\\test_62\\rewardings.txt")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_62\\2819_0")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_62\\1805_0")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_71\\9316_0")
# env = BoudaryEnv(boundary())
# env = BoudaryEnv(read_polygon('../ui/domains/dolphine2.json'))
# env.boundary.show()
# env = BoudaryEnv(read
# _ = env.reset()
# env.render()
# env.close()
# check_env(env, warn=True)
# env = BoudaryEnv(read_polygon('../ui/domains/test1.json'))
# env.boundary.show(style='k.-')
# env.boundary.segmts_diff_ratio()
