import torch
from .dq import *



def dq_adjoint(v: torch.Tensor, dq: torch.Tensor):
    """
    Adjoint for a dual quaternion dq tensor of shape [*, 8] and a twist v [*, 6].

    In:
        - v: [B, 6]: tensor with a set of twists
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z,  w_,x_,y_,z_)
    
    Out: 
        - adj: [B, 6]: tensor with the adjoint
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

    In: 
        - dq_: [B, 6]: tensor with a set of twists from the dual quaternion group

    Out: 
        - res: [B, 8]: tensor with the corresponding dual quaternions (w,x,y,z,  w_,x_,y_,z_)
    """

    assert dq_.shape[-1] == 6

    device = dq_.device
    
    # Add real part to the primary and dual part (converts it to a pure dual quaternion)
    dq_ = torch.cat((torch.zeros((dq_.shape[0], 1), device = device), dq_[:, :3], 
                     torch.zeros((dq_.shape[0], 1), device = device), dq_[:, 3:]), dim = -1)

    # Norm of the primary part
    phi = torch.norm(dq_[:, :4], dim = -1)
    phi_idx = phi != 0

    # Identity quaternion
    prim = torch.cat((torch.ones(dq_.shape[0], 1), torch.zeros(dq_.shape[0], 3)), dim = -1).to(device)
    
    # Assign the identity or apply the exponential mapping wether ...
    # ... the primary part is null or not
    prim[phi_idx] = (torch.sin(phi[phi_idx]) / phi[phi_idx]).unsqueeze(-1) * dq_[phi_idx, :4]
    prim[phi_idx, 0] += torch.cos(phi[phi_idx])

    # Concatenates the primary with the dual part
    res = torch.cat((prim, q_mul(dq_[:, 4:], prim)), dim = -1)

    # Fix dual cover
    neg_idx = res[:, 0] < 0.0
    res[neg_idx] *= -1
    
    return res



def log_bruno(dq: torch.Tensor):
    """"
    Logarithmic map for a twist dq_ of shape [*, 6].

    In:
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z,  w_,x_,y_,z_)

    Out:
        - res: [B, 6]: tensor with the corresponding twsits
    """

    assert dq.shape[-1] == 8

    # Fix dual cover
    # neg_idx = dq[:, 0] < 0.0
    # dq[neg_idx] *= -1

    # Converts the parts
    primary = (q_angle(q = dq[:, :4]).unsqueeze(-1)*0.5)*q_axis(q = dq[:, :4])

    print(primary)
    print(dq)
    print(q_angle(q = dq[:, :4]))
    print(q_axis(q = dq[:, :4]))
    print("\n\n")

    dual = dq_translation(dq = dq)*0.5

    # Returns the concatenation of both parts
    return torch.cat((primary, dual), dim = -1)


def exp_stereo_q(q_: torch.Tensor):
    """
    Stereographic exponential map for a pure quaternion of shape [*, 3].

    In: 
        - q_: [B, 3]: tensor with a set of pure quaternions without the real part (x,y,z)
    
    Out: 
        - res: [B, 4]: tensor with the corresponding unit quaternions (w,x,y,z)
    """

    assert q_.shape[-1] == 3

    # Denominator
    div = (1 + torch.norm(q_, dim = -1)).unsqueeze(-1)

    # Real part and imaginary part
    re = (1 - torch.pow(torch.norm(q_, dim = -1), 2)).unsqueeze(-1) / div
    im = 2*q_ / div

    # Return the concatenation of both parts
    return torch.cat((re, im), dim = -1)


def log_stereo_q(q: torch.Tensor):
    """
    Stereographic logarithmic map for a unit quaternion of shape [*, 4].

    In: 
        - q: [B, 4]: tensor with a set of unit quaternions (w,x,y,z)
    
    Out: 
        - res: [B, 3]: tensor with the corresponding pure quaternions (x,y,z)
    """
    assert q.shape[-1] == 4

    return q[:, 1:] / (1+q[:, 0]).unsqueeze(-1)


def exp_stereo(dq_:torch.Tensor, kwargs = None):
    """
    Stereographic exponential map for a dual quaternion of shape [*, 8] -> it is the same as the map for a quaternion+translation.

    In: 
        - dq_: [B, 6]: tensor with a set of twists from the dual quaternion group
    
    Out: 
        - res: [B, 8]: tensor with the corresponding unit dual quaternions quaternions (w,x,y,z, w_,x_,y_,z_)
    """

    assert dq_.shape[-1] == 6
    
    # Obtain the exponential of the primary part with the stereographic map and ...
    q_p = exp_stereo_q(q_ = dq_[:, :3])  

    # ... the translation from the dual part
    res = dq_from_tr(t = dq_[:, 3:], r = q_p)

    # Fix dual cover
    neg_idx = res[:, 0] < 0.0
    res[neg_idx] *= -1

    return res


def log_stereo(dq: torch.Tensor):
    """
    Stereographic logarithmic map for a dual quaternion of shape [*, 8] -> is the same as the map for a quaternion+translation.

    In: 
        - dq: [B, 8]: tensor with a set of unit dual quaternions (w,x,y,z, w_,x_,y_,z_)
    
    Out: 
        - res: [B, 6]: tensor with the corresponding twists
    """

    assert dq.shape[-1] == 8
    assert torch.any(dq_is_norm(dq))

    # Fix dual cover
    neg_idx = dq[:, 0] < 0.0
    dq[neg_idx] *= -1

    # Concatenate the logarithm of the primary part with the translation
    return torch.cat((log_stereo_q(q = dq[:, :4]), 
                      dq_translation(dq = dq)), dim = -1)

