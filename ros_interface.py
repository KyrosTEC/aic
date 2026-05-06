# ros_interface.py

import numpy as np


class AICRosInterface:
    def __init__(self, node):
        self.node = node

    def reset_episode(self):
        """
        First version:
        Do nothing.
        
        Later:
        respawn board/cable, randomize task, tare force sensor during training.
        """
        pass

    def get_observation_vector(self):
        """
        Return the vector PPO sees.
        First version can use:
        - TCP pose
        - TCP velocity
        - force
        - plug-to-port delta from ground truth during training
        """
        obs = self.get_latest_observation()

        tcp = obs.controller_state.tcp_pose
        vel = obs.controller_state.tcp_velocity
        wrench = obs.wrist_wrench.wrench

        dx, dy, dz = self.get_plug_to_port_delta()

        return np.array([
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
        ], dtype=np.float32)

    def move_tcp_delta(self, dx, dy, dz):
        """
        Send a small MotionUpdate to the robot.
        Similar idea to FirstPolicy:
        current_pose + small delta.
        """
        raise NotImplementedError

    def get_latest_observation(self):
        raise NotImplementedError

    def get_plug_to_port_delta(self):
        """
        During training, use ground truth TF.
        During final evaluation, you cannot rely on this.
        """
        raise NotImplementedError

    def get_plug_to_port_distance(self):
        dx, dy, dz = self.get_plug_to_port_delta()
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def get_force_norm(self):
        obs = self.get_latest_observation()
        f = obs.wrist_wrench.wrench.force
        return float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

    def is_inserted(self):
        """
        First version:
        distance < small threshold.

        Later:
        use contact/insertion signal if available during training.
        """
        return self.get_plug_to_port_distance() < 0.005