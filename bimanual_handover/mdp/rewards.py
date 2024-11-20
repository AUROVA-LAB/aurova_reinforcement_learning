# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from .dq import dq_distance, q_mul


# Compute error as a dual quaternion distance
def dual_quaternion_error(pose1: torch.Tensor, pose2: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose1 / pose2 - torch.Tensor(N, 7): poses to compute the distance in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - distance - torch.Tensor(N, 3): distance[:, 0] in dual quaternions, translation module[:, 1] and rotation module[:, 2].
    '''
    # Convert position and orientation (pose) to dual quaternion
    pose1_dq = pose2dq(pose = pose1, device = device)
    pose2_dq = pose2dq(pose = pose2, device = device)

    # Compute dual quaternion distance
    distance = dq_distance(pose2_dq, pose1_dq)
        
    return distance


##
# UTILS
##

# Transforms a pose / frame into a dual quaternion
def pose2dq(pose: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pose - torch.Tensor(N, 7): pose to be converted into a dual quaternion in translation(3) + rotation in quaternions(4).
        - device - str: Device into which the environment is stored.

    Out:
        - return - torch.Tensor(N, 8): frame represented as a dual quaternion.
    '''

    # Separate position and orientation
    pos = pose[:, :3]
    orient = pose[:, 3:]

    # Converts position to a simple quaternion
    pos = trans2q(pos = pos, device = device)

    # Shape comprobation
    assert pos.shape[-1] == 4
    assert orient.shape[-1] == 4

    # From translation and orientation to DQ
    return torch.cat((orient, 0.5 * q_mul(orient, pos)), dim = -1)


# Transforms a linear translation to simple quaternion
def trans2q(pos: torch.Tensor, device: str) -> torch.Tensor:
    '''
    In:
        - pos - torch.Tensor(N, 3): position to be converted into a simple quaternion.
        - device - str: Device into which the environment is stored.

    Out:
        - return - torch.Tensor(N, 4): translation represented as a simple quaternion.
    '''

    # Shape comprobation
    assert pos.shape[-1] == 3

    # Add zero to the imaginary part of the translation quaternion
    return torch.cat((torch.zeros((pos.size()[0], 1)).to(device), pos), dim = -1)