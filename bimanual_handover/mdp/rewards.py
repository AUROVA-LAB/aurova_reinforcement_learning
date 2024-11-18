# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_error_magnitude, quat_mul

from .dq import dq_distance, q_mul

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Gets the asset: the robot

    command = env.command_manager.get_command(command_name)
    # Gets the command in the local frame

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    # Position part from the command in the local frame

    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    # Converts the desired pose in the robot frame to the world frame
    # asset.data.root_state_w --> Robot base/root position in world frame
    # des_pos_b --> frame of the desired pose in local frame -> frame that is transformed

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # Obtains the current position in the world frame 
    # asset.data.body_state_w --> obtains the world pose of all bodies of the asset. 
    #      This command obtains the world position of the selected body (ee_link)
    # asset_cfg.body_ids --> list of body IDs passed as numbers


    return torch.norm(curr_pos_w - des_pos_w, dim=1)
    # Gets the position error

# THIS FUNCTION IS SIMILAR TO THE PREVIOUS ONE, although the hyperbolic function is applied to limit the reward
def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Gets the asset: the robot

    command = env.command_manager.get_command(command_name)
    # Gets the command in the local frame

    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    # Orientation part of the command in the local frame

    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    # Transforms the desired obervation in the local frame to the global frame
    # asset.data.root_state_w -> Robot base/root position in world frame

    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    # Obtains the current orientation of the selected body in the global frame

    return quat_error_magnitude(curr_quat_w, des_quat_w)
    # Error computation usign quaternions

def dual_quaternion_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Gets the asset: the robot

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:,:3]
    des_or_b = command[:, 3:7]
    # Gets the command in the local frame
    # command --> frame of the desired pose in local frame

    curr_pose_w = asset.data.body_state_w[:,asset_cfg.body_ids[0], :7]
    
    # Obtains the current position in the world frame 

    base_pose = asset.data.root_state_w[:, :7]
    # Pose of the base
    # asset.data.root_state_w --> Robot base/root position in world frame

    # des_pos_w, des_or_w = combine_frame_transforms(base_pose[:,:3], base_pose[:,:7], command[:,:3], command[:,:7])
    # Converts the desired pose in the robot frame to the world frame

    curr_pos_b, curr_or_b = subtract_frame_transforms(base_pose[:,:3], base_pose[:,3:7], curr_pose_w[:,:3], curr_pose_w[:,3:7])
    # Converts the current position in the world frame to the robot frame
    # The current is in the world frame. Then it has to be converted to the base frame, so the ...
    #    ... inverse of the base frame is used.
    # It is the inverse operation of the "position_command_error" function.

    curr_pos_b = trans2q(pos = curr_pos_b, device = env.device)
    des_pos_b = trans2q(pos = des_pos_b, device = env.device)
    # Convert position to simple quaternion

    curr_dq = pose2dq(pos = curr_pos_b, orient = curr_or_b)
    des_dq = pose2dq(pos = des_pos_b, orient = des_or_b)
    # Convert position and orientation (pose) to dual quaternion

    distance, translation_mod, orientation_mod = dq_distance(des_dq, curr_dq)
    # Compute dual quaternion distance
        
    return distance


##
# UTILS
##

def pose2dq(pos: torch.tensor, orient: torch.tensor) -> torch.tensor:

    # Shape comprobation
    assert pos.shape[-1] == 4
    assert orient.shape[-1] == 4

    # From translation and orientation to DQ
    return torch.cat((orient, 0.5 * q_mul(orient, pos)), dim = 1)

def trans2q(pos: torch.tensor, device: str) -> torch.tensor:

    # Shape comprobation
    assert pos.shape[-1] == 3

    # Add zero to the imaginary part of the translation quaternion
    return torch.cat((torch.zeros((pos.size()[0], 1)).to(device), pos), dim=1)