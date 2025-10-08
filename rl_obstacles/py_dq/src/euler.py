import torch
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

# =========== EULER ANGLES with TRANSLATION = [x,y,z,roll,pitch,yaw] =================

def convert_euler_to_Lab(x: torch.Tensor):
    """
    Convert a Translation+Euler to the format of IsaacLab (translation+quaternion).
    """
    assert x.shape[-1] == 6

    e = quat_from_euler_xyz(roll = x[:, 3], pitch = x[:, 4], yaw = x[:, 5])

    return torch.cat((x[:, :3], e), dim = -1)

def from_quat_to_euler(t: torch.Tensor, r: torch.Tensor):
    """
    Convert to Translation+Quaternion to Translation+Euler.
    """
    assert t.shape[-1] == 3
    assert r.shape[-1] == 4

    neg_idx = r[:, 0] < 0.0
    r[neg_idx] *= -1

    r,p,y = euler_xyz_from_quat(quat = r)
    e = torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim = -1)

    return torch.cat((t, e), dim = -1)

def euler_diff(x1: torch.Tensor, x2: torch.Tensor):
    """
    Difference between two Translation+Euler x1 and x2 of shape [*, 6].
    """
    assert x1.shape[-1] == 6
    assert x2.shape[-1] == 6

    return x2 - x1

def euler_mul(x1: torch.Tensor, x2: torch.Tensor):
    """
    Addition between two Translation+Euler x1 and x2 of shape [*, 6].
    """
    assert x1.shape[-1] == 6
    assert x2.shape[-1] == 6

    return x1 + x2

def euler_normalize(x: torch.Tensor):
    """
    Normalize the Translation+Euler representation of shape [*, 6] -> No change.
    """
    assert x.shape[-1] == 6

    return x

def identity_map(x):
    """
    Performs identity mapping.
    """
    return x
