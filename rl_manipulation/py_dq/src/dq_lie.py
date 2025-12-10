import torch
from .dq import *



def dq_adjoint(v: torch.Tensor, dq: torch.Tensor):
    """
    Adjoint for a dual quaternion dq tensor of shape [*, 8] and a twist v [*, 6].
    """
    assert v.shape[-1] == 6
    assert dq.shape[-1] == 8

    device = dq.device
    
    v = torch.cat((torch.zeros(v.shape[0], 1).to(device), v[:, :3],
                   torch.zeros(v.shape[0], 1).to(device), v[:, 3:]), dim = -1)
    
    adj = dq_mul(dq1 = dq, dq2 = dq_mul(dq1 = v, dq2 = dq_conjugate(dq = dq)))

    return torch.cat((adj[:, 1:4], adj[:, 5:]), dim = -1)



def exp_bruno(dq_: torch.Tensor, kwargs = None):
    """"
    Exponential map for a twist dq_ of shape [*, 6].
    """

    assert dq_.shape[-1] == 6

    device = dq_.device

    dq_ = torch.cat((torch.zeros((dq_.shape[0], 1), device = device), dq_[:, :3], 
                     torch.zeros((dq_.shape[0], 1), device = device), dq_[:, 3:]), dim = -1)

    phi = torch.norm(dq_[:, :4], dim = -1)
    phi_idx = phi != 0

    prim = torch.cat((torch.ones(dq_.shape[0], 1), torch.zeros(dq_.shape[0], 3)), dim = -1).to(device)
    prim[phi_idx] = (torch.sin(phi[phi_idx]) / phi[phi_idx]).unsqueeze(-1) * dq_[phi_idx, :4]
    prim[phi_idx, 0] += torch.cos(phi[phi_idx])

    res = torch.cat((prim, q_mul(dq_[:, 4:], prim)), dim = -1)

    neg_idx = res[:, 0] < 0.0
    res[neg_idx] *= -1
    
    return res

def log_bruno(dq: torch.Tensor):
    """"
    Logarithmic map for a twist dq_ of shape [*, 6].
    """
    assert dq.shape[-1] == 8

    neg_idx = dq[:, 0] < 0.0
    dq[neg_idx] *= -1

    primary = (q_angle(q = dq[:, :4]).unsqueeze(-1)*0.5)*q_axis(q = dq[:, :4])
    dual = dq_translation(dq = dq)*0.5

    return torch.cat((primary, dual), dim = -1)


def exp_stereo_q(q_: torch.Tensor):
    """
    Stereographic exponential map for a pure quaternion of shape [*, 3].
    """
    assert q_.shape[-1] == 3
    div = (1 + torch.norm(q_, dim = -1)).unsqueeze(-1)

    re = (1 - torch.pow(torch.norm(q_, dim = -1), 2)).unsqueeze(-1) / div
    im = 2*q_ / div

    return torch.cat((re, im), dim = -1)

def log_stereo_q(q: torch.Tensor):
    """
    Stereographic logarithmic map for a unit quaternion of shape [*, 4].
    """
    assert q.shape[-1] == 4

    return q[:, 1:] / (1+q[:, 0]).unsqueeze(-1)

def exp_stereo(dq_:torch.Tensor, kwargs = None):
    """
    Stereographic exponential map for a dual quaternion of shape [*, 8] -> is the same as the map for a quaternion+translation.
    """
    assert dq_.shape[-1] == 6
    
    q_p = exp_stereo_q(q_ = dq_[:, :3])  
    dq = dq_from_tr(t = dq_[:, 3:], r = q_p)

    neg_idx = dq[:, 0] < 0.0
    dq[neg_idx] *= -1

    return dq

def log_stereo(dq: torch.Tensor):
    """
    Stereographic logarithmic map for a dual quaternion of shape [*, 8] -> is the same as the map for a quaternion+translation.
    """
    assert dq.shape[-1] == 8
    assert torch.any(dq_is_norm(dq))

    neg_idx = dq[:, 0] < 0.0
    dq[neg_idx] *= -1

    t_ = dq_translation(dq = dq)

    return torch.cat((log_stereo_q(q = dq[:, :4]), t_), dim = -1)

