import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from robotless_env import *
from networks import *

# ---------------------------------------------------------
# 1. Create multiple environments
# ---------------------------------------------------------

def make_env():
    return BasicPoseEnv()


n_envs = 1 # 64

env = make_vec_env(
    make_env,
    n_envs=n_envs,
)


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
    device="auto",
)


# ---------------------------------------------------------
# 4. Train
# ---------------------------------------------------------

model.learn(
    total_timesteps=100_000,
)