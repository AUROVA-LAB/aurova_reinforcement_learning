import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import wandb
from datetime import datetime
import os
from wandb.integration.sb3 import WandbCallback

from robotless_env import *
from networks import *



path_to_train = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation/train"
log_dir = os.path.join(path_to_train, "logs", "sb3", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
run = wandb.init(project="rl_manipulation_reach", name=log_dir.split("/")[-1], sync_tensorboard=True)



# ---------------------------------------------------------
# 1. Create multiple environments
# ---------------------------------------------------------

def make_env():
    return BasicPoseEnv()


n_envs = 64

env = make_vec_env(
    make_env,
    n_envs=n_envs,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000 // n_envs,
    save_path="./checkpoints/",
    name_prefix="ppo_robotless",
)

callback = CallbackList([
    checkpoint_callback,

    WandbCallback(
        gradient_save_freq=1000,
        model_save_path=None,
        verbose=2,
    ),
])


# ---------------------------------------------------------
# 2. Custom network
# ---------------------------------------------------------

policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(
        features_dim=128,
    ),
    net_arch=dict(
        pi=[128, 64],
        vf=[128, 64],
    ),
)


# ---------------------------------------------------------
# 3. PPO
# ---------------------------------------------------------

model = PPO(
    policy="MlpPolicy",
    env=env,

    policy_kwargs=policy_kwargs,

    n_steps=256,
    batch_size=1024,

    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,

    ent_coef=0.01,
    clip_range=0.2,

    verbose=1,
    device="cpu",
)


# ---------------------------------------------------------
# 4. Train
# ---------------------------------------------------------

model.learn(
    total_timesteps=100_000,
    callback=callback
)