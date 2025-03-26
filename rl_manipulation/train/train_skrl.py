# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "SAC", "DDPG", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

import skrl
from packaging import version

import gymnasium as gym

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from train_utils import *


# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # set the environment seed
    # note: certain randomization occur in the environment initialization so we set the seed here
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # Key for the "directory" key in the configuration file
    dir_string = "base_directory" if args_cli.algorithm == "SAC" else "directory"

    # specify directory for logging experiments
    #log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"][dir_string])
    #log_root_path = os.path.abspath(log_root_path)
    #print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    #log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    #if agent_cfg["agent"]["experiment"]["experiment_name"]:
    #    log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    #agent_cfg["agent"]["experiment"][dir_string] = log_root_path
    #agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    #log_dir = os.path.join(log_root_path, log_dir)

    path_to_train = "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train"
    log_dir = os.path.join(path_to_train, "logs", "skrl", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):# and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # Training for PPO
    if args_cli.algorithm == "PPO":
        print("\n\n --- Perforimg PPO")
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        agent_cfg['agent']['experiment']['directory'] = log_dir
        agent_cfg['agent']['experiment']['experiment_name'] = log_dir.split("/")[-1]
        agent_cfg['agent']['experiment']['wandb_kwargs'] = {"project": "bim_hand_dani_julio",
                                                            "name": log_dir.split("/")[-1],
                                                            "sync_tensorboard": True}
        print(agent_cfg['agent']['experiment'])
        runner = Runner(env, agent_cfg)

        # # run training
        runner.run()

    # Training for SAC
    elif args_cli.algorithm == "SAC":
        print("\n\n --- Perforimg SAC")
        device = env.device

        # instantiate a memory as experience replay
        memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

        # instantiate the agent's models (function approximators).
        # SAC requires 5 models, visit its documentation for more details
        # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
        models = {}
        models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
        models["critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["critic_2"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

        # initialize models' parameters (weights and biases)
        for model in models.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

        agent = SAC(models=models,
                memory=memory,
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 15000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

        # start training
        trainer.train()

    else:
        raise ValueError(f"Sorry, {args_cli.algorithm} agents are not currently supported.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()