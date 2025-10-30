import math
import torch
from .dq_lie import exp_stereo_q, log_stereo_q, q_is_norm, q_diff, q_mul, q_normalize, q_norm, q_conjugate


# ======== TRANSLATION + QUATERNION: [x,y,z,w,x_,y_,z_] =======================================================================

def convert_quat_trans_to_Lab(q: torch.Tensor):
    """
    IsaacLab works with translation + quaternion.
    """
    assert q.shape[-1] == 7

    return q

def identity_map_conversion(x: torch.Tensor, q: torch.Tensor):
    """"
    Identity mapping.
    """

    return torch.cat((x, q), dim = -1)


def norm_quat(x: torch.Tensor):
    """"
    Normalize the Translation+Quaternion representation of shape [*, 7].
    """
    assert x.shape[-1] == 7

    x[:, 3:] = q_normalize(q = x[:, 3:])

    return x


def q_trans_diff(q1: torch.Tensor, q2: torch.Tensor):
    """
    Difference between two Translation+Quaternion q1 and q2 of shape [*, 7].
    """
    assert q1.shape[-1] == 7
    assert q2.shape[-1] == 7
    
    # Translation
    t_diff = q2[:, :3] - q1[:, :3]
    
    # Rotation
    r_diff = q_diff(q1 = q1[:, 3:], q2 = q2[: , 3:])

    return torch.cat((t_diff, r_diff), dim = -1)

def q_trans_mul(q1: torch.Tensor, q2: torch.Tensor):
    """
    Multiplication between two Translation+Quaternion q1 and q2 of shape [*, 7].
    """
    assert q1.shape[-1] == 7
    assert q2.shape[-1] == 7

    t = q1[:, :3] + q2[:, :3]
    r = q_mul(q1 = q1[:, 3:], q2 = q2[:, 3:])

    return torch.cat((t, r), dim = -1)




def exp_quat_stereo(q_:torch.Tensor):
    """
    Stereographic exponential map for a Translation+Quaternion of shape [*, 7].
    """
    assert q_.shape[-1] == 6

    device = q_.device

    q_p = exp_stereo_q(q_ = q_[:, 3:])    
    q_p = q_normalize(q = q_p)

    return torch.cat((q_[:, :3], q_p), dim = -1).to(device)

def log_quat_stereo(q: torch.Tensor):
    """
    Stereographic logarithmic map for a Translation+Quaternion of shape [*, 7].
    """
    assert q.shape[-1] == 7

    device = q.device

    t_ = q[:, :3]

    return torch.cat((t_, log_stereo_q(q = q[:, 3:])), dim = -1).to(device)

