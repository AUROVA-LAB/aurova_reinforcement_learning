import torch
from .lab_utils import euler_xyz_from_quat, quat_from_euler_xyz

# =========== EULER ANGLES with TRANSLATION = [x,y,z,roll,pitch,yaw] =================

def convert_euler_to_Lab(x: torch.Tensor):
    """
    Convert a Translation+Euler to the format of IsaacLab (translation+quaternion).
    
    In: 
        - x: [B, 6]: tensor with a set of translation+euler angles (x,y,z, roll,pitch,yaw)

    Out:
        - res: [B, 7]: tensor with the corresponding translation+quaternion (x,y,z, w,x,y,z)
    """

    assert x.shape[-1] == 6

    # Obtainer the quaternion from the euler angles
    e = quat_from_euler_xyz(roll = x[:, 3], pitch = x[:, 4], yaw = x[:, 5])

    return torch.cat((x[:, :3], e), dim = -1)


def from_quat_to_euler(t: torch.Tensor, r: torch.Tensor):
    """
    Convert to Translation+Quaternion to Translation+Euler.

    In: 
        - t: [B, 3]: tensor with a set of translation (roll,pitch,yaw)
        - r: [B, 4]: tensor with a set of euler angles (x,y,z)

    Out:
        - res: [B, 7]: tensor with the corresponding translation+euler (x,y,z, roll,pitch,yaw)
    """

    assert t.shape[-1] == 3
    assert r.shape[-1] == 4

    # Fix double cover
    neg_idx = r[:, 0] < 0.0
    r[neg_idx] *= -1

    # Obtain the euler angles from the quaternion
    r,p,y = euler_xyz_from_quat(quat = r)
    e = torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim = -1)

    # Concatenates the translation with the euler angles
    return torch.cat((t, e), dim = -1)


def euler_diff(x1: torch.Tensor, x2: torch.Tensor):
    """
    Difference between two Translation+Euler x1 and x2 of shape [*, 6].

    In: 
        - x1: [B, 6]: tensor with a set of translation+euler (x,y,z, roll,pitch,yaw)
        - x2: [B, 6]: tensor with another set of translation+euler (x,y,z, roll,pitch,yaw)

    Out:
        - res: [B, 6]: tensor with the difference between both tensors
    """

    assert x1.shape[-1] == 6
    assert x2.shape[-1] == 6

    return x2 - x1


def euler_mul(x1: torch.Tensor, x2: torch.Tensor):
    """
    Addition between two Translation+Euler x1 and x2 of shape [*, 6].

    In: 
        - x1: [B, 6]: tensor with a set of translation+euler (x,y,z, roll,pitch,yaw)
        - x2: [B, 6]: tensor with another set of translation+euler (x,y,z, roll,pitch,yaw)

    Out:
        - res: [B, 6]: tensor with the multiplication/addition between both tensors
    """

    assert x1.shape[-1] == 6
    assert x2.shape[-1] == 6

    return x1 + x2


def euler_normalize(x: torch.Tensor):
    """
    Normalize the Translation+Euler representation of shape [*, 6] -> No change.

    In: 
        - x: [B, 6]: tensor with a set of translation+euler (x,y,z, roll,pitch,yaw)

    Out:
        - res: [B, 6]: tensor with the normalisation (it is the same as the input)
    """
    assert x.shape[-1] == 6

    return x


def identity_map(x):
    """
    Performs identity mapping.

    In:
        - x: [B, D]: tensor with elements

    Out:
        - res: [B, D]: the same tensor as the input
    """

    return x
