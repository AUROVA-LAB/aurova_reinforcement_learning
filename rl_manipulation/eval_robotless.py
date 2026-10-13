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


def make_env():
    return BasicPoseEnv()

"""Main function"""
def main():

    
    # Environment creation
    
    env = make_vec_env(
        make_env,
        n_envs=1,
    )
    
    # Filter models
    path_to_train = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation/checkpoints"

    dir = os.path.join(path_to_train, "2026-10-08_09-38-39")

    models = [file for file in os.listdir(dir) if file.endswith(".zip")]

    # Reset
    obs = env.reset()

    # --- Loop through the models ---
    for idx, model_name in enumerate(models):
        
        model_name_ = "rl_manipulation_reach_99968_steps.zip"

        # Loading model
        model = PPO.load(os.path.join(dir, model_name_))
        model.policy.eval()

        print(f"\n\n{idx + 1}. Loading model: " + model_name_)

        # Simulate physics
        while True:
            with torch.inference_mode():
                # Generate action
                action, __ = model.predict(obs, deterministic = True)

                # Step the environemnt
                obs, rew, done, info = env.step(torch.tensor(action))
                
                print(rew)

            if done:
                env.reset()
                print("--- RESET ---")
                raise
                       
                    

    env.close()


if __name__ == "__main__":
    # Run the main the function
    main()

    # Close sim app
    simulation_app.close()