# train_ppo.py
"""
Train a first PPO residual policy for the AIC cable insertion task.

Run this from the AIC repo root, with the simulator already running in another
terminal, for example:

    distrobox enter -r aic_eval -- /entrypoint.sh \
      ground_truth:=true \
      start_aic_engine:=false \
      spawn_task_board:=true \
      spawn_cable:=true \
      attach_cable_to_gripper:=true \
      cable_type:=sfp_sc_cable

Then:

    pixi run python training/train_ppo.py \
      --port-frame task_board/nic_card_mount_0/sfp_port_0_link \
      --plug-frame sfp_sc_cable/sfp_module_link

Important:
- This is TRAINING code, not final evaluation code.
- The port/plug TF frames use ground truth and must not be used directly in the
  final submitted policy.
- This script assumes ros_interface.py implements AICRosInterface with a real
  rclpy Node, Observation subscriber, MotionUpdate publisher, and TF lookup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ppo_env import AICPPOEnv
from ros_interface import AICRosInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for AIC local insertion.")

    parser.add_argument(
        "--port-frame",
        type=str,
        required=True,
        help=(
            "Ground-truth target port frame for TRAINING reward, e.g. "
            "task_board/nic_card_mount_0/sfp_port_0_link"
        ),
    )
    parser.add_argument(
        "--plug-frame",
        type=str,
        required=True,
        help=(
            "Ground-truth plug frame for TRAINING reward, e.g. "
            "sfp_sc_cable/sfp_module_link or sfp_sc_cable/sc_plug_link"
        ),
    )
    parser.add_argument(
        "--observation-topic",
        type=str,
        default="/aic_model/observation",
        help="Observation topic. Check with: ros2 topic list | grep -i observation",
    )
    parser.add_argument(
        "--pose-command-topic",
        type=str,
        default="/aic_controller/pose_commands",
        help="Cartesian MotionUpdate topic.",
    )
    parser.add_argument(
        "--base-frame",
        type=str,
        default="base_link",
        help="Base frame used for TCP targets and TF lookup.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="training/checkpoints",
        help="Where to save checkpoints and final model.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="training/logs",
        help="Where to save monitor logs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_insert_policy",
        help="Final model name without .zip extension.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Max environment steps per episode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = rclpy.create_node("aic_ppo_training_node")

    try:
        node.get_logger().info("Creating AICRosInterface...")
        ros_interface = AICRosInterface(
            node=node,
            observation_topic=args.observation_topic,
            pose_command_topic=args.pose_command_topic,
            base_frame=args.base_frame,
            port_frame=args.port_frame,
            plug_frame=args.plug_frame,
        )

        node.get_logger().info("Waiting for first observation and TF frames...")
        ros_interface.reset_episode()

        env = AICPPOEnv(ros_interface, max_steps=args.max_steps)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))

        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_aic_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.98,
            tensorboard_log=str(log_dir),
        )

        node.get_logger().info(f"Starting PPO training for {args.total_timesteps} timesteps...")
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

        final_path = checkpoint_dir / args.model_name
        model.save(str(final_path))
        node.get_logger().info(f"Saved final PPO model to {final_path}.zip")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

