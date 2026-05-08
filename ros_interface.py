# ros_interface.py
"""
ROS <-> Gym/PPO bridge for AIC training.

This file is intended for TRAINING, not final evaluation. In this first version,
`get_plug_to_port_delta()` uses ground-truth TF frames so PPO can receive a
supervised target error / reward signal while learning.

Typical use:
    Terminal 1:
        distrobox enter -r aic_eval -- /entrypoint.sh \
          ground_truth:=true start_aic_engine:=false \
          spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true

    Terminal 2:
        pixi run python training/train_ppo.py

Before training, check your actual observation topic:
    ros2 topic list | grep -i observation
If needed, pass observation_topic="<topic>" to AICRosInterface.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Vector3, Wrench
from tf2_ros import Buffer, TransformException, TransformListener

from aic_model_interfaces.msg import Observation
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode


class AICRosInterface:
    """Small ROS interface used by ppo_env.py.

    This class:
      - subscribes to an Observation topic,
      - publishes Cartesian MotionUpdate commands,
      - uses TF ground truth to compute plug-to-port delta during training.

    Notes:
      - It assumes the simulator is running with `ground_truth:=true`.
      - It assumes the controller is in Cartesian target mode.
      - It does NOT reset/randomize the world yet; `reset_episode()` is a no-op.
    """

    def __init__(
        self,
        node: Node,
        *,
        observation_topic: str = "/aic_model/observation",
        pose_command_topic: str = "/aic_controller/pose_commands",
        base_frame: str = "base_link",
        port_frame: Optional[str] = None,
        plug_frame: Optional[str] = None,
        command_dt: float = 0.08,
    ):
        if node is None:
            raise ValueError(
                "AICRosInterface needs a real rclpy Node. "
                "Create one in train_ppo.py with rclpy.create_node(...)."
            )

        self.node = node
        self.base_frame = base_frame
        self.port_frame = port_frame
        self.plug_frame = plug_frame
        self.command_dt = command_dt

        self.latest_observation: Optional[Observation] = None

        self.obs_sub = self.node.create_subscription(
            Observation,
            observation_topic,
            self._observation_callback,
            10,
        )

        self.pose_pub = self.node.create_publisher(
            MotionUpdate,
            pose_command_topic,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.node.get_logger().info(
            f"AICRosInterface ready. Subscribing to {observation_topic}, "
            f"publishing to {pose_command_topic}."
        )

    # ------------------------------------------------------------------
    # ROS callbacks / waiting helpers
    # ------------------------------------------------------------------
    def _observation_callback(self, msg: Observation) -> None:
        self.latest_observation = msg

    def spin_some(self, timeout_sec: float = 0.01) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def wait_for_first_observation(self, timeout_sec: float = 10.0) -> Observation:
        start = time.monotonic()
        while self.latest_observation is None:
            self.spin_some(0.05)
            if time.monotonic() - start > timeout_sec:
                raise TimeoutError(
                    "No Observation received. Check the topic name with: "
                    "ros2 topic list | grep -i observation"
                )
        return self.latest_observation

    def wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.node.get_clock().now()
        timeout = Duration(seconds=timeout_sec)
        while (self.node.get_clock().now() - start) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            try:
                self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                pass
        self.node.get_logger().warn(
            f"TF not available: {source_frame} -> {target_frame}. "
            "Are you running with ground_truth:=true and correct frame names?"
        )
        return False

    # ------------------------------------------------------------------
    # Methods used by ppo_env.py
    # ------------------------------------------------------------------
    def reset_episode(self):
        """First version: no world reset/randomization.

        For the first PPO experiment, launch one scenario, manually move/script
        the robot close to the port, then let PPO learn local corrections.

        Later, replace this with:
          - delete/spawn task board/cable,
          - randomize task configuration,
          - move robot to a consistent start pose,
          - tare F/T sensor if available in training.
        """
        self.wait_for_first_observation(timeout_sec=10.0)
        if self.port_frame and self.plug_frame:
            self.wait_for_tf(self.base_frame, self.port_frame, timeout_sec=5.0)
            self.wait_for_tf(self.base_frame, self.plug_frame, timeout_sec=5.0)

    def get_latest_observation(self) -> Observation:
        """Return the latest Observation message, spinning ROS if needed."""
        if self.latest_observation is None:
            return self.wait_for_first_observation(timeout_sec=10.0)

        # Keep subscriptions fresh.
        self.spin_some(0.001)
        return self.latest_observation

    def get_observation_vector(self) -> np.ndarray:
        """Return the vector PPO sees.

        Current first-training observation:
          [tcp_x, tcp_y, tcp_z,
           tcp_vx, tcp_vy, tcp_vz,
           force_x, force_y, force_z,
           target_dx, target_dy, target_dz]

        The last 3 values are from ground-truth TF and must later be replaced
        by a legal perception estimate for final evaluation.
        """
        obs = self.get_latest_observation()

        tcp = obs.controller_state.tcp_pose
        vel = obs.controller_state.tcp_velocity
        wrench = obs.wrist_wrench.wrench

        dx, dy, dz = self.get_plug_to_port_delta()

        return np.array(
            [
                tcp.position.x,
                tcp.position.y,
                tcp.position.z,
                vel.linear.x,
                vel.linear.y,
                vel.linear.z,
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                dx,
                dy,
                dz,
            ],
            dtype=np.float32,
        )

    def move_tcp_delta(self, dx: float, dy: float, dz: float) -> None:
        """Publish a small Cartesian pose target: current TCP pose + delta.

        This keeps the current TCP orientation unchanged and only changes x/y/z.
        The target is sent as a MotionUpdate to /aic_controller/pose_commands.
        """
        obs = self.get_latest_observation()
        current_pose = obs.controller_state.tcp_pose
        target_pose = deepcopy(current_pose)
        target_pose.position.x += float(dx)
        target_pose.position.y += float(dy)
        target_pose.position.z += float(dz)

        motion_update = MotionUpdate(
            header=Header(
                frame_id=self.base_frame,
                stamp=self.node.get_clock().now().to_msg(),
            ),
            pose=target_pose,
            target_stiffness=np.diag([45.0, 45.0, 35.0, 25.0, 25.0, 25.0]).flatten(),
            target_damping=np.diag([30.0, 30.0, 28.0, 14.0, 14.0, 14.0]).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

        self.pose_pub.publish(motion_update)

        # Let the robot/controller and subscriptions advance a little.
        end_time = time.monotonic() + self.command_dt
        while time.monotonic() < end_time:
            self.spin_some(0.005)

    def get_plug_to_port_delta(self) -> Tuple[float, float, float]:
        """Return port_position - plug_position using ground-truth TF.

        Training only. Do not use this method in final submitted evaluation
        logic unless the frames are part of the allowed observation interface.
        """
        if not self.port_frame or not self.plug_frame:
            raise RuntimeError(
                "port_frame and plug_frame must be provided for ground-truth reward. "
                "Example: port_frame='task_board/nic_card_mount_0/sfp_port_0_link', "
                "plug_frame='sfp_sc_cable/sfp_module_link'."
            )

        try:
            port_tf = self.tf_buffer.lookup_transform(self.base_frame, self.port_frame, Time())
            plug_tf = self.tf_buffer.lookup_transform(self.base_frame, self.plug_frame, Time())
        except TransformException as exc:
            raise RuntimeError(
                f"Could not read ground-truth TF frames. port_frame={self.port_frame}, "
                f"plug_frame={self.plug_frame}. Are you running ground_truth:=true?"
            ) from exc

        p = port_tf.transform.translation
        q = plug_tf.transform.translation

        return (float(p.x - q.x), float(p.y - q.y), float(p.z - q.z))

    def get_plug_to_port_distance(self) -> float:
        dx, dy, dz = self.get_plug_to_port_delta()
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    def get_force_norm(self) -> float:
        obs = self.get_latest_observation()
        f = obs.wrist_wrench.wrench.force
        return float(math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

    def is_inserted(self) -> bool:
        """First approximation for training termination.

        This is not the official insertion sensor. It only says: if the plug
        tip is very close to the target port frame, treat it as success for the
        first PPO experiment.
        """
        return self.get_plug_to_port_distance() < 0.005

