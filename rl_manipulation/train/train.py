# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=7500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--train", type=bool, default=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=200_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime
import torch

import omni
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from source.isaaclab_tasks.isaaclab_tasks.manager_based.aurova_reinforcement_learning.rl_manipulation.train.networks import *

# PLACEHOLDER: Extension template (do not remove this comment)

import wandb
from wandb.integration.sb3 import WandbCallback

from contextlib import redirect_stdout
from pathlib import Path



# directory for logging into
path_to_train = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation/train"
log_dir = os.path.join(path_to_train, "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
run = wandb.init(project="rl_manipulation", name=log_dir.split("/")[-1], sync_tensorboard=True)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        raise
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    # log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    policy_dict = {"activation_fn": torch.nn.Tanh, "net_arch": [64, 128, 256], "squash_output": True, "share_features_extractor": True, }


    agent = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir, seed = 14, n_steps = 16, batch_size = 4096, gae_lambda = 0.95, gamma = 0.99, n_epochs = 20, ent_coef = 0.01, learning_rate = 1e-04, clip_range = 0.2, use_sde = True, policy_kwargs = policy_dict, vf_coef = 1.0, max_grad_norm = 1.0, device = 'cuda')
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval), WandbCallback()]


    # train the agent
    if args_cli.train:
        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(
                total_timesteps=n_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=None,
            )
        # save the final model
        agent.save(os.path.join(log_dir, "model"))
        print("Saving to:")
        print(os.path.join(log_dir, "model.zip"))

        if isinstance(env, VecNormalize):
            print("Saving normalization")
            env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # see the environment
    else:
        obs = env.reset()
                             
        action = torch.zeros((env_cfg.scene.num_envs, env_cfg.size))
        action = torch.tensor([[0,0,0,0,0,1]]).repeat(env_cfg.scene.num_envs, 1)

        # Simulate physics
        while True:
            with torch.inference_mode():

                # Step the environment
                ret = env.step(action)

        n_samples = 256
        samples = []
        isaac_env = env.unwrapped

        T_tcp_initial = np.array([-0.4919,  0.1333,  0.4879, 3.2050e-06, -3.8268e-01, -9.2388e-01,  2.0429e-06])
        
        T_initial_torch = torch.tensor(T_tcp_initial).view(-1).unsqueeze(0)
        T_initial_group = isaac_env.convert_to_group(T_initial_torch[:, :3], T_initial_torch[:, 3:])
        T_initial_se3 = isaac_env.to_group_dist(t = T_initial_torch[:, :3], r = T_initial_torch[:, 3:])
        T_initial_se3 = T_initial_se3.view(-1, 4, 4).squeeze(0).cpu().numpy()
        T_initial_inv = np.linalg.inv(T_initial_se3)

        T_initial_lie = isaac_env.log(T_initial_group.float()).cpu().numpy()
        T_initial_lie = np.repeat(T_initial_lie, env.num_envs, axis=0)

        policy = agent.policy
        policy.set_training_mode(False)
        obs_tensor, _ = policy.obs_to_tensor(obs)


        raw_samples, sent_samples, delta_samples = [], [], []


        with torch.no_grad():
            for _ in range(256):
                policy.reset_noise(env.num_envs)
                action_tensor, _, _ = policy(obs_tensor, deterministic=False)

                raw_action = action_tensor.cpu().numpy()

                sent_action = action_tensor.cpu().clone().numpy()


                sent_action[:, :3] *= isaac_env.cfg.action_scaling[0]
                sent_action[:, 3:] *= isaac_env.cfg.action_scaling[1]

                raw_action[:, :3] *= isaac_env.cfg.action_scaling[0]
                raw_action[:, 3:] *= isaac_env.cfg.action_scaling[1]

                # Sustituye esta llamada por tu decodificador MAT o DQ.
                sent_action = np.clip(
                    raw_action, agent.action_space.low, agent.action_space.high
                )

                if isaac_env.cfg.mapping == 0:
                    if isaac_env.cfg.representation == isaac_env.cfg.DQ:
                        sent_action = torch.tensor(sent_action)
                        iden = torch.zeros_like(sent_action)
                        iden[:, 0] = 1.0
                        sent_action += iden

                        T_target_group = isaac_env.mul_operator(T_initial_group.repeat(env.num_envs, 1).float(), sent_action.float())

                        sent_action = sent_action.cpu().numpy()

                    elif isaac_env.cfg.representation == isaac_env.cfg.QUAT:
                        sent_action = torch.tensor(sent_action)
                        iden = torch.zeros_like(sent_action[:, 3:])
                        iden[:, 0] = 1.0
                        sent_action[:, 3:] += iden

                        T_target_group = isaac_env.mul_operator(T_initial_group.repeat(env.num_envs, 1).float(), sent_action.float())

                        sent_action = sent_action.cpu().numpy()

                    elif isaac_env.cfg.representation == isaac_env.cfg.EULER:
                        sent_action = torch.tensor(sent_action)
                        
                        T_target_group = isaac_env.mul_operator(T_initial_group.repeat(env.num_envs, 1).float(), sent_action.float())

                        sent_action = sent_action.cpu().numpy()
                    
                    
                else:
                    T_target_lie = T_initial_lie + sent_action

                    T_target_lie_torch = torch.tensor(T_target_lie)
                    T_target_group = isaac_env.exp(T_target_lie_torch)

                T_target_lab = isaac_env.convert_to_Lab(T_target_group)
                T_target = isaac_env.to_group_dist(t = T_target_lab[:, :3], r = T_target_lab[:, 3:])
                T_target = T_target.view(-1, 4, 4).squeeze(0).cpu().numpy()


                raw_samples.append(raw_action.copy())
                sent_samples.append(sent_action.copy())
                delta_samples.append(T_initial_inv @ T_target)
    
    report_path = Path("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation") / Path("experimento_1_DQ_seed" + str(agent_cfg["seed"]) + ".txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report_file:
        with redirect_stdout(report_file):
            print_experiment_1(
                "DQ | escala original | seed " + str(agent_cfg["seed"]),
                np.stack(delta_samples),
                np.stack(raw_samples),
                np.stack(sent_samples),
            )

    print(f"Informe guardado en {report_path.resolve()}")
    
    # close the simulator
    env.close()


from scipy.spatial.transform import Rotation


def print_experiment_1(
    name,
    delta_T,
    raw_actions,
    sent_actions,
    *,
    near_zero_mm=0.1,
    near_zero_deg=0.1,
    clip_atol=1e-7,
):
    """Resume incrementos del TCP y clipping para acciones Box sin squash.

    delta_T:       (n_samples, n_envs, 4, 4), T_tcp_initial^-1 @ T_tcp_target
    raw_actions:   (n_samples, n_envs, action_dim), antes del clip de SB3
    sent_actions:  (n_samples, n_envs, action_dim), después del clip de SB3
    """
    delta_T = np.asarray(delta_T, dtype=np.float64)
    raw_actions = np.asarray(raw_actions)
    sent_actions = np.asarray(sent_actions)

    if delta_T.ndim != 4 or delta_T.shape[-2:] != (4, 4):
        raise ValueError("delta_T debe tener forma (n_samples, n_envs, 4, 4)")
    if (
        raw_actions.ndim != 3
        or raw_actions.shape != sent_actions.shape
        or raw_actions.shape[:2] != delta_T.shape[:2]
    ):
        raise ValueError("Las acciones y delta_T deben compartir muestras y entornos")

    n_samples, n_envs = delta_T.shape[:2]
    delta_T = delta_T.reshape(-1, 4, 4)
    dp_mm = delta_T[:, :3, 3] * 1_000.0
    rotvec_deg = Rotation.from_matrix(delta_T[:, :3, :3]).as_rotvec(degrees=True)

    distance_mm = np.linalg.norm(dp_mm, axis=1)
    angle_deg = np.linalg.norm(rotvec_deg, axis=1)

    clip_error = np.abs(raw_actions - sent_actions)
    clipped = clip_error > clip_atol
    almost_static = (distance_mm <= near_zero_mm) & (angle_deg <= near_zero_deg)

    def pearson(x, y):
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    print(f"\n=== Experimento 1 | {name} ===")
    print(f"{n_samples} muestras × {n_envs} entornos = {len(distance_mm)} acciones")
    print(f"{'Magnitud':<22} {'Media':>8} {'DE':>8} {'P05':>8} "
          f"{'P50':>8} {'P95':>8} {'P99':>8} {'Máx.':>8}")

    for label, values in (
        ("Traslación [mm]", distance_mm),
        ("Giro [deg]", angle_deg),
    ):
        percentiles = np.percentile(values, [5, 50, 95, 99])
        stats = [values.mean(), values.std(), *percentiles, values.max()]
        print(f"{label:<22}" + "".join(f" {value:8.3f}" for value in stats))

    fmt = {"precision": 3, "suppress_small": True}
    print("\nPor eje, en el marco inicial del TCP [x, y, z]")
    print("Traslación media [mm] :", np.array2string(dp_mm.mean(axis=0), **fmt))
    print("Traslación DE [mm]    :", np.array2string(dp_mm.std(axis=0), **fmt))
    print("Rotvec medio [deg]   :", np.array2string(rotvec_deg.mean(axis=0), **fmt))
    print("Rotvec DE [deg]      :", np.array2string(rotvec_deg.std(axis=0), **fmt))

    corr = pearson(distance_mm, angle_deg)
    corr_label = f"{corr:.3f}" if np.isfinite(corr) else "indefinida (varianza nula)"
    print(f"\nPearson entre magnitudes de traslación y giro = {corr_label}")

    cross_corr = np.array([
        [pearson(dp_mm[:, i], rotvec_deg[:, j]) for j in range(3)]
        for i in range(3)
    ])
    print("Pearson por eje (filas dx, dy, dz | columnas rx, ry, rz)")
    print(np.array2string(cross_corr, **fmt))

    print(f"\nAcciones con algún canal recortado = {100 * clipped.any(axis=-1).mean():.2f}%")
    print("Clipping por canal [%] =", np.array2string(
        100 * clipped.mean(axis=(0, 1)), **fmt
    ))
    if clipped.any():
        print(f"Recorte medio en canales afectados = {clip_error[clipped].mean():.4f}")
    print(f"Incrementos casi nulos (≤ {near_zero_mm:g} mm y ≤ {near_zero_deg:g} deg)"
          f" = {100 * almost_static.mean():.2f}%")

import math
import torch



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()