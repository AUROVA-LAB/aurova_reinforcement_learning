import math
import torch
from .lie import exp_stereo_q, log_stereo_q, exp_stereo_t, log_stereo_t, q_is_norm, q_diff, q_mul, q_normalize, q_norm, q_conjugate


def convert_quat_trans_to_Lab(q: torch.Tensor):
    assert q.shape[-1] == 7
    assert torch.any(q_is_norm(q[:, 3:]))

    return q


def norm_quat(x: torch.Tensor):
    return x


def q_trans_diff(q1: torch.Tensor, q2: torch.Tensor):
    assert q1.shape[-1] == 7
    assert q2.shape[-1] == 7
    assert torch.any(q_is_norm(q1[:, 3:]))
    assert torch.any(q_is_norm(q2[:, 3:]))

    t_diff = q2[:, :3] - q1[:, :3]
    r_diff = q_diff(q1 = q1[:, 3:], q2 = q2[: , 3:])

    return torch.cat((t_diff, r_diff), dim = -1)

def q_trans_mul(q1: torch.Tensor, q2: torch.Tensor):
    assert q1.shape[-1] == 7
    assert q2.shape[-1] == 7
    assert torch.any(q_is_norm(q1[:, 3:]))
    assert torch.any(q_is_norm(q2[:, 3:]))

    t = q1[:, :3] + q2[:, :3]
    r = q_mul(q1 = q1[:, 3:], q2 = q2[:, 3:])

    return torch.cat((t, r), dim = -1)



def exp_quat_jil(q_: torch.Tensor):
    assert q_.shape[-1] == 6

    device = q_.device

    w = q_[:, 3:]
    t = q_[:, :3]

    norm = torch.norm(w, dim = -1)

    re_w = torch.cos(norm / 2)
    im_w = (torch.sin(norm / 2) / norm).unsqueeze(-1) * w

    exp_w = torch.cat((re_w.unsqueeze(-1), im_w), dim = -1)

    idx_0 = torch.abs(norm) < 1e-6
    exp_w[idx_0] = torch.Tensor([1.0, 0.0, 0.0, 0.0]).repeat(q_.shape[0], 1).to(device)[idx_0]

    return torch.cat((t, exp_w), dim = -1)

def log_quat_jil(q: torch.Tensor):
    assert q.shape[-1] == 7
    assert torch.any(q_is_norm(q[:, 3:]))

    w = q[:, 3:]
    t = q[:, :3]

    re_w = w[:, 0]
    im_w = w[:, 1:]

    w = 2 * (torch.acos(re_w) / torch.norm(im_w, dim = -1)).unsqueeze(-1) * im_w

    return torch.cat((t, w), dim = -1) 



def exp_quat_stereo(q_:torch.Tensor):
    assert q_.shape[-1] == 6

    q_p = exp_stereo_q(q_ = q_[:, 3:])    
    q_p = q_normalize(q = q_p)

    return torch.cat((q_[:, :3], q_p), dim = -1)

def log_quat_stereo(q: torch.Tensor):
    assert q.shape[-1] == 7
    assert torch.any(q_is_norm(q[:, 3:]))

    t_ = q[:, :3]

    return torch.cat((t_, log_stereo_q(q = q[:, 3:])), dim = -1)



def exp_quat_cayley(q_: torch.Tensor):
    assert q_.shape[-1] == 6
    
    t = q_[:, :3]
    q_ = torch.cat((torch.zeros(q_.shape[0], 1), q_[:, 3:]), dim = -1)
    
    identity = torch.cat((torch.ones(q_.shape[0], 1), torch.zeros(q_.shape[0], 3)), dim = -1)
    denom = q_norm(identity - q_[:, 3:]).unsqueeze(-1)

    res = q_mul(q1 = (identity + q_), q2 = q_conjugate(identity - q_) / (denom*denom) )
    
    return torch.cat((t, q_normalize(res)), dim = -1)

def log_quat_cayley(q: torch.Tensor):
    assert q.shape[-1] == 7
    assert torch.any(q_is_norm(q = q[:, 3:]))

    identity = torch.cat((torch.ones(q.shape[0], 1), torch.zeros(q.shape[0], 3)), dim = -1)
    denom = q_norm(q[:, 3:] + identity).unsqueeze(-1)

    res = q_mul(q1 = q[:, 3:] - identity, q2 = q_conjugate(q[:, 3:] + identity) / (denom*denom) )

    return torch.cat((q[:, :3], res[:, 1:4]), dim = -1)







# t1 = torch.tensor([[-4.9190e-01,  1.3330e-01,  4.8790e-01]]).repeat(2,1)
# r1 = torch.tensor([[3.3840e-06, -3.8268e-01, -9.2388e-01,  2.1003e-06]]).repeat(2,1)

# t2 = torch.tensor([[-0.1,  0.1348,  0.3480]]).repeat(2,1)
# r2 = torch.tensor([[0.2521, 0.0346, 0.9422,  -0.2179]]).repeat(2,1)

# t2 = torch.tensor([[-0.0,  0.01,  0.0]]).repeat(2,1)
# r2 = torch.tensor([[0.5, -0.866, 0.0,  -0.0]]).repeat(2,1)


# q1 = torch.cat((t1, r1), dim = -1)
# q2 = torch.cat((t2, r2), dim = -1)


# x = q1
# q = q2
# print("Q1: ", x.round(decimals = 4))
# print("Q2: ", q.round(decimals = 4))


# q2[:, 3:] = q_normalize(q2[:, 3:])
# x_q = q_trans_diff(q1 = x, q2 = q)
# x_q[:, 3:] = q_normalize(x_q[:, 3:])

# print("DIFF: ", x_q.round(decimals = 4))

# l_ = log_quat_jil(q = x_q)
# b = exp_quat_jil(q_ = l_)

# print("DIFF: ", b.round(decimals = 4))
# print("PRE Logaritmico: ", l_.round(decimals = 4))

# b = q_trans_mul(x, b)
# b[:, 3:] = q_normalize(b[:, 3:])


# print("Logaritmico: ", l_.round(decimals = 4))
# print("Exponencial: ", b.round(decimals = 4))