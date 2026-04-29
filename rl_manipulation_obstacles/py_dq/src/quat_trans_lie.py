import math
import torch
from .dq_lie import exp_stereo_q, log_stereo_q, q_diff, q_mul, q_normalize


# ======== TRANSLATION + QUATERNION: [x,y,z,w,x_,y_,z_] =======================================================================

def convert_quat_trans_to_Lab(q: torch.Tensor):
    """
    Convert to itself

    In: 
        - q: [B, 7]: tensor with a set of translation+quaternion (x,y,z, w,x,y,z)
    Out:
        - res: [B, 7]: tensor with the same translatio+quaternion as the input
    """

    assert q.shape[-1] == 7

    return q

def identity_map_conversion(x: torch.Tensor, q: torch.Tensor):
    """"
    Identity mapping

    In:
        - x: [B, 3]: tensor with a set of translations (x,y,z)
        - q: [B, 7]: tensor with a set of rotations as quaternions (w,x,y,z)
    Out:
        - res: [B, 7]: tensor with the inputs concatenated
    """

    return torch.cat((x, q), dim = -1)


def norm_quat(x: torch.Tensor):
    """"
    Normalize the Translation+Quaternion representation of shape [*, 7]

    In:
        - x: [B, 7]: tensor with a set of translation+quaternion (x,y,z, w,x,y,z)
    Out:
        - res: [B, 7]: tensor with the rotation normalized
    """

    assert x.shape[-1] == 7

    # Normalizes only the quaternion
    x[:, 3:] = q_normalize(q = x[:, 3:])

    return x


def q_trans_diff(q1: torch.Tensor, q2: torch.Tensor):
    """
    Difference between two Translation+Quaternion q1 and q2 of shape [*, 7]
    
    In: 
        - q1: [B, 7]: tensor with a set of translation+quaternion (x,y,z, w,x,y,z)
        - q2: [B, 7]: tensor with another set of translation+quaternion (x,y,z, w,x,y,z)
    Out:
        - res: [B, 7]: tensor with the difference between both inputs
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
    Multiplication between two Translation+Quaternion q1 and q2 of shape [*, 7]

    In: 
        - q1: [B, 7]: tensor with a set of translation+quaternion (x,y,z, w,x,y,z)
        - q2: [B, 7]: tensor with another set of translation+quaternion (x,y,z, w,x,y,z)
    Out:
        - res: [B, 7]: tensor with the product of both inputs
    """

    assert q1.shape[-1] == 7
    assert q2.shape[-1] == 7

    t = q1[:, :3] + q2[:, :3]
    r = q_mul(q1 = q1[:, 3:], q2 = q2[:, 3:])

    return torch.cat((t, r), dim = -1)




def exp_quat_stereo(q_:torch.Tensor):
    """
    Stereographic exponential map for a Translation+Quaternion of shape [*, 7]

    In: 
        - q_: [B, 6]: tensor with a set of twists from translation+quaternion 
    Out:
        - res: [B, 7]: tensor with its corresponding translation+quaternion (x,y,z, w,x,y,z) 
    """

    assert q_.shape[-1] == 6

    device = q_.device

    # Map only rotation
    q_p = exp_stereo_q(q_ = q_[:, 3:])    
    q_p = q_normalize(q = q_p)

    # Concatenantes the translation with the rotation
    return torch.cat((q_[:, :3], q_p), dim = -1).to(device)


def log_quat_stereo(q: torch.Tensor):
    """
    Stereographic logarithmic map for a Translation+Quaternion of shape [*, 7]

    In: 
        - q_: [B, 7]: tensor with a set of translation+quaternion (x,y,z, w,x,y,z) 
    Out:
        - res: [B, 6]: tensor with its corresponding twists
    """
    
    assert q.shape[-1] == 7

    device = q.device

    t_ = q[:, :3]

    # Concatenates the translation with the rotation
    return torch.cat((t_, log_stereo_q(q = q[:, 3:])), dim = -1).to(device)

