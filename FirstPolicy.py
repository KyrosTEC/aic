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


import numpy as np


from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration


class FirstPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("FirstPolicy.__init__()")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback("DEBUG InsertCable started - movimiento brusco")
        self.get_logger().info(f"DEBUG insert_cable() enter. Task: {task}")

        obs = get_observation()

        if obs is None:
            send_feedback("No initial observation received")
            self.get_logger().info("No initial observation received.")
            return False

        start_pose = self._copy_pose(obs.controller_state.tcp_pose)

        # Movimientos GRANDES para comprobar que el robot sí se mueve.
        # Ajusta estos valores si son demasiado agresivos para tu setup.
        debug_offsets = [
            (0.00,  0.00,  0.04),
            (0.08,  0.00,  0.04),
            (-0.08, 0.00,  0.04),
            (0.00,  0.08,  0.04),
            (0.00, -0.08,  0.04),
            (0.06,  0.06,  0.02),
            (-0.06, 0.06,  0.02),
            (0.06, -0.06,  0.02),
            (-0.06, -0.06, 0.02),
            (0.00,  0.00,  0.00),
        ]

        for i, (dx, dy, dz) in enumerate(debug_offsets):
            obs = get_observation()

            if obs is None:
                send_feedback(f"Step {i}: no observation")
                self.get_logger().info(f"Step {i}: no observation received.")
                continue

            current_force = self._force_norm(obs)

            send_feedback(
                f"DEBUG step {i}: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, force={current_force:.2f}N"
            )

            self.get_logger().info(
                f"DEBUG moving step {i}: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, force={current_force:.2f}N"
            )

            target_pose = self._copy_pose(start_pose)
            target_pose.position.x += dx
            target_pose.position.y += dy
            target_pose.position.z += dz

            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=[90, 90, 90, 45, 45, 45],
                damping=[35, 35, 35, 18, 18, 18],
            )

            self.sleep_for(0.8)

        # Regresar a la posición inicial
        send_feedback("DEBUG returning to start pose")

        self.set_pose_target(
            move_robot=move_robot,
            pose=start_pose,
            stiffness=[80, 80, 80, 40, 40, 40],
            damping=[35, 35, 35, 18, 18, 18],
        )

        self.sleep_for(1.0)

        send_feedback("DEBUG InsertCable finished")
        self.get_logger().info("DEBUG insert_cable() exiting.")
        return True

