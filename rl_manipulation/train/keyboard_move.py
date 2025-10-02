import argparse
# from pynput import keyboard

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="UR5e RL environment")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-UR5e-joint-reach-v0", help="Name of the task.")  # Isaac-UR5e-reach-v0

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows"""

import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
import gymnasium as gym
from pynput import keyboard

# Key commands
key_cmd = '\n'
pos_cmd = "sdp869"
neg_cmd = "wal241"
inc = 1

# Action variables
action = torch.zeros(6)
incs = torch.zeros_like(action)


# Press callback
def on_press(key):
    global key_cmd, incs

    try:
        incs *= 0
        key_cmd = key.char
        # incs = update_cmd(incs)
        
    except AttributeError:
        pass

# Update the command according to the key
def update_cmd(cmd):
    global key_cmd, pos_cmd, neg_cmd, inc

    if key_cmd in pos_cmd:
        list_cmds, mult = pos_cmd, inc

    elif key_cmd in neg_cmd:
        list_cmds, mult = neg_cmd, -inc

    else: 
        return cmd

    idx = list_cmds.index(key_cmd)

    key_cmd = '\n'

    cmd[:, idx] += mult

    return cmd



"""Main function"""
def main():
    global action, incs

    # Batch actions
    action = action.repeat(args_cli.num_envs, 1)
    incs = incs.repeat(args_cli.num_envs, 1)
    
    # Start keyboard listener
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Environment
    env_cfg = parse_env_cfg(
        task_name = args_cli.task, device = args_cli.device, num_envs = args_cli.num_envs, use_fabric = not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg = env_cfg)
    
    # Reset
    env.reset()

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():

            obs, rew, terminated, truncated, info = env.step(update_cmd(action))



    env.close()


if __name__ == "__main__":
    # Run the main the function
    main()

    # Close sim app
    simulation_app.close()