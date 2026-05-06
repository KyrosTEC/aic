# train_ppo.py

from stable_baselines3 import PPO
from ppo_env import AICPPOEnv
from ros_interface import AICRosInterface

# You need to create/start a ROS node here.
# This part depends on your ROS setup.
ros_interface = AICRosInterface(node=None)

env = AICPPOEnv(ros_interface)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.98,
)

model.learn(total_timesteps=100_000)
model.save("ppo_insert_policy")