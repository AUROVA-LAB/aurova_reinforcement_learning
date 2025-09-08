import torch
from .dq import *
from .lie import *


# === DQ DISTANCES =================================================================================

def dqLOAM_distance(dq_pred: torch.Tensor, dq_real: torch.Tensor, log_fc = None, diff_fc = None) -> torch.Tensor:
    '''
    Calculates the screw motion parameters between dual quaternion representations of the given poses pose_pred/real.
    The difference of two dual quaternions results the pure dual quaternion if the originals represent the same frame in space.
    => "Distance" between two dual quaternions: how different they are to the pure dual quaternion.
    '''

    # Obtain the difference between two dual quaternion
    res = dq_mul(dq_real, dq_conjugate(dq_pred))

    # If the result was the pure dual quaternion, the real part of the dual part would be 1 and the rest 0s.
    #    The norm of that modified dual quaternion is 0. --> This is not geometrically correct because ...
    #    ... the module of the simple quaternions and the whole dual must be unitary but this method serves as a way to compute distances.
    res[:, 0] = torch.abs(res[:, 0]) - 1

    # Obtain the norm of the primary and dual part
    translation_mod = torch.norm(res[:, 4:], dim = -1)
    rotation_mod = torch.norm(res[:, :4], dim = -1)

    # The distance is the sum of the modules
    distance = translation_mod + rotation_mod

    # return torch.cat((distance, translation_mod, rotation_mod)).transpose(0,1)
    return distance


# === Lie DISTANCES =================================================================================

def geodesic_dist(x1: torch.Tensor, x2: torch.Tensor, log_fc = None, diff_fc = None) -> torch.Tensor:

    log_diff = log_fc(diff_fc(x1, x2))[0]

    return torch.norm(log_diff, dim = -1)


def double_geodesic_dist(x1: torch.Tensor, x2: torch.Tensor, log_fc = None, diff_fc = None) -> torch.Tensor:

    log_x1 = log_fc(x1)[0]
    log_x2 = log_fc(x2)[0]

    return q_inn_prod(q1 = log_x1[:, :3], q2 = log_x2[:, :3]) + q_inn_prod(q1 = log_x1[:, 3:], q2 = log_x2[:, 3:])