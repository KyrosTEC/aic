#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
FirstPolicy_scripted.py

A safer scripted baseline for the AIC qualification task.

Important:
- This policy does NOT use ground-truth TF frames, /gz_server, /scoring, or simulator internals.
- It assumes the robot starts close to the target, as described in the qualification phase.
- It tries to earn more than model-validity points by ending near the port entrance or partially inserted.
- It is still a blind/local-search policy, not a learned RL policy.

Strategy:
1. Read the initial TCP pose.
2. Keep the initial orientation, because the challenge starts the robot close to the target.
3. Move gently through a small local search pattern around the start pose.
4. At each local-search point, attempt a slow downward insertion.
5. If force rises too much, back off slightly and try the next nearby offset.
6. Do NOT return to the start pose, because scoring uses final plug proximity/insertion.
"""

import math
from copy import deepcopy

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion


class FirstPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("FirstPolicy scripted baseline initialized")

        # Conservative force threshold. Scoring penalizes sustained >20 N, so stay below it.
        self.force_stop_n = 12.0

        # Motion timing. The controller smooths commands internally, but we still send small steps.
        self.command_dt = 0.08

        # Cartesian stiffness/damping. Lower during insertion = more compliant.
        self.free_space_stiffness = [70, 70, 70, 35, 35, 35]
        self.free_space_damping = [35, 35, 35, 16, 16, 16]
        self.insert_stiffness = [45, 45, 35, 25, 25, 25]
        self.insert_damping = [30, 30, 28, 14, 14, 14]

    # -------------------------------------------------------------------------
    # Small utilities
    # -------------------------------------------------------------------------
    def _copy_pose(self, pose: Pose) -> Pose:
        return deepcopy(pose)

    def _pose_with_offset(self, base: Pose, dx: float, dy: float, dz: float) -> Pose:
        p = self._copy_pose(base)
        p.position.x += dx
        p.position.y += dy
        p.position.z += dz
        return p

    def _pose_with_offset_negative(self, base: Pose, dx: float, dy: float, dz: float) -> Pose:
        p = self._copy_pose(base)
        p.position.x -= dx
        p.position.y -= dy
        p.position.z -= dz
        return p

    def _force_norm(self, obs: Observation) -> float:
        try:
            f = obs.wrist_wrench.wrench.force
            return float(math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
        except Exception:
            return 0.0

    def _move_and_wait(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        wait_s: float,
        stiffness: list[float],
        damping: list[float],
    ) -> None:
        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=stiffness,
            damping=damping,
        )
        self.sleep_for(wait_s)

    def _gentle_insert_attempt(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        base_pose: Pose,
        dx: float,
        dy: float,
        max_depth: float,
        step: float,
    ) -> bool:
        """Try a slow vertical insertion at one local-search XY offset.

        Returns True if the attempt reached full requested depth without high force.
        Returns False if force got too high and we backed off.
        """
        depth = 0.0
        last_pose = self._pose_with_offset_negative(base_pose, dx, dy, 0.0)

        while depth > -max_depth:
            obs = get_observation()
            if obs is None:
                send_feedback("No observation during insertion attempt")
                return False

            force = self._force_norm(obs)
            if force > self.force_stop_n:
                send_feedback(
                    f"Force backoff: {force:.1f} N at dx={dx:.3f}, dy={dy:.3f}, depth={depth:.3f}"
                )
                # Back off upward by 5 mm and stop this attempt.
                backoff = self._copy_pose(last_pose)
                backoff.position.z -= 0.005
                self._move_and_wait(
                    move_robot,
                    backoff,
                    wait_s=0.20,
                    stiffness=self.insert_stiffness,
                    damping=self.insert_damping,
                )
                return False

            last_pose = self._pose_with_offset_negative(base_pose, dx, dy, depth)
            self._move_and_wait(
                move_robot,
                last_pose,
                wait_s=self.command_dt,
                stiffness=self.insert_stiffness,
                damping=self.insert_damping,
            )

            depth -= step

        return True

    # -------------------------------------------------------------------------
    # Main policy
    # -------------------------------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback("FirstPolicy scripted insertion started")
        self.get_logger().info(f"FirstPolicy scripted insert_cable() task: {task}")

        obs = get_observation()
        if obs is None:
            send_feedback("No initial observation received")
            return False

        start_pose = self._copy_pose(obs.controller_state.tcp_pose)

        # The qualification docs say the robot starts within a few centimeters of the target.
        # So we do a local search around the initial pose instead of large waving motions.
        # Keep final pose near the best attempted insertion, do NOT return to start.

        # 1) Stabilize / slight lift to reduce accidental contact before searching.
        safe_pose = self._pose_with_offset_negative(start_pose, 0.0, 0.0, 0.010)
        send_feedback("Stage 1: stabilize above start pose")
        for _ in range(8):
            self._move_and_wait(
                move_robot,
                safe_pose,
                wait_s=0.05,
                stiffness=self.free_space_stiffness,
                damping=self.free_space_damping,
            )

        # 2) Local search pattern in XY. Small offsets avoid task-board collisions.
        # These values are intentionally small because the start is already close.
        search_offsets = [
            (0.000, 0.000),
            (0.04, 0.000), (-0.04, 0.000), (0.000, 0.04), (0.000, -0.04),
            (0.06, 0.06), (-0.06, 0.06), (0.06, -0.06), (-0.06, -0.06),
            (0.10, 0.000), (-0.10, 0.000), (0.000, 0.10), (0.000, -0.10),
        ]

        # SFP tends to need slightly smaller motions; SC can tolerate a bit more.
        plug_name = str(getattr(task, "plug_name", "")).lower()
        port_name = str(getattr(task, "port_name", "")).lower()
        is_sc = ("sc" in plug_name) or ("sc" in port_name)

        max_depth = 0.45 if is_sc else 0.35
        step = 0.0012

        send_feedback(
            f"Stage 2: local search + insertion attempts; is_sc={is_sc}, max_depth={max_depth:.3f}"
        )

        best_final_pose = safe_pose
        for idx, (dx, dy) in enumerate(search_offsets):
            obs = get_observation()
            if obs is None:
                send_feedback(f"Search {idx}: no observation")
                continue

            force = self._force_norm(obs)
            if force > self.force_stop_n:
                send_feedback(f"High force before search {idx}: {force:.1f} N; lifting")
                lift = self._pose_with_offset_negative(safe_pose, dx, dy, 0.008)
                self._move_and_wait(
                    move_robot,
                    lift,
                    wait_s=0.25,
                    stiffness=self.insert_stiffness,
                    damping=self.insert_damping,
                )
                continue

            send_feedback(f"Search {idx}: dx={dx:.3f}, dy={dy:.3f}")

            # Move above this search offset first.
            above = self._pose_with_offset_negative(safe_pose, dx, dy, 0.0)
            self._move_and_wait(
                move_robot,
                above,
                wait_s=0.25,
                stiffness=self.free_space_stiffness,
                damping=self.free_space_damping,
            )

            # Try to insert slowly.
            reached_depth = self._gentle_insert_attempt(
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                base_pose=safe_pose,
                dx=dx,
                dy=dy,
                max_depth=max_depth,
                step=step,
            )

            # Keep track of a useful final pose. If we reached depth, stay there.
            best_final_pose = self._pose_with_offset_negative(safe_pose, dx, dy, -max_depth)
            if reached_depth:
                send_feedback(f"Reached full insertion depth at search {idx}; holding pose")
                break

        # 3) Hold final pose so the scoring sensors can register partial/full insertion.
        send_feedback("Stage 3: holding final attempted insertion pose")
        for _ in range(40):
            obs = get_observation()
            if obs is not None and self._force_norm(obs) > self.force_stop_n:
                # Tiny backoff if force rises during hold.
                best_final_pose.position.z -= 0.002
            self._move_and_wait(
                move_robot,
                best_final_pose,
                wait_s=0.05,
                stiffness=self.insert_stiffness,
                damping=self.insert_damping,
            )

        send_feedback("FirstPolicy scripted insertion finished")
        self.get_logger().info("FirstPolicy scripted insert_cable() exiting")
        return True
