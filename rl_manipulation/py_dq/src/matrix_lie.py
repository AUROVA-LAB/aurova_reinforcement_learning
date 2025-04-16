import math
import torch
from .matrix import *
from .dq import q_is_norm
from omni.isaac.lab.utils.math import quat_from_matrix




def convert_homo_to_Lab(x: torch.Tensor):
    assert x.shape[-1] == 16

    x = x.view(-1, 4, 4)
    q = quat_from_matrix(matrix = x[:, :-1, :-1])

    return torch.cat((x[:, :-1, -1], q), dim = -1)





def log_mat(mat: torch.tensor):
    assert mat.shape[-1] == 16

    device = mat.device

    mat = mat.view(-1, 4, 4)
    t = mat[:, :-1, -1]
    mat = mat[:, :-1, :-1]

    theta = torch.acos(0.5 * (torch.vmap(torch.trace)(mat) - 1))

    log = (theta / (2*torch.sin(theta)) * torch.stack([mat[:, 2,1] - mat[:, 1,2], 
                                                       mat[:, 0,2] - mat[:, 2,0],
                                                       mat[:, 1,0] - mat[:, 0,1]])).transpose(0,1)
    idx_0 = torch.abs(theta) < 1e-6
    idx_pi = torch.isclose(theta, torch.tensor(math.pi))


    log[idx_0] = torch.zeros_like(log)[idx_0]

    S = mat + mat.transpose(-1, -2) + (1 - torch.vmap(torch.trace)(mat))[:, None, None]*torch.eye(3).to(device)
    S = torch.stack((S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]), dim = -1)
    n = S / (3 - torch.vmap(torch.trace)(mat))[:, None]

    R = torch.stack((mat[:, 0, 0], mat[:, 1, 1], mat[:, 2, 2]), dim = -1)
    log[idx_pi] = (n * torch.sqrt(0.5*(1 + R)))[idx_pi]

    return torch.cat((log, t), dim = -1)

def exp_mat(mat_: torch.tensor):
    assert mat_.shape[-1] == 6

    device = mat_.device

    theta = torch.norm(mat_[:, :3], dim = -1)

    n = mat_[:, :3] / theta.unsqueeze(-1)
    
    c = torch.cos
    s  = torch.sin

    # exp = torch.sin(theta) / theta *  mat_[:, :3] + (1 - torch.cos(theta)) / theta**2 * mat_[:, :3]**2

    r1 = torch.stack([c(theta) + n[:, 0]**2 * (1-c(theta)),                     
                       n[:, 0] * n[:, 1] * (1 - c(theta)) - n[:, 2]*s(theta),    
                       n[:, 0] * n[:, 2] * (1 - c(theta)) + n[:, 1]*s(theta)], dim = -1)
    r2 = torch.stack([n[:, 0] * n[:, 1] * (1 - c(theta)) + n[:, 2]*s(theta),    
                      c(theta) + n[:, 1]**2 * (1-c(theta)),                     
                      n[:, 1] * n[:, 2] * (1 - c(theta)) - n[:, 0]*s(theta)], dim = -1)
    r3 = torch.stack([n[:, 0] * n[:, 2] * (1 - c(theta)) - n[:, 1]*s(theta),    
                      n[:, 1] * n[:, 2] * (1 - c(theta)) + n[:, 0]*s(theta),    
                      c(theta) + n[:, 2]**2 * (1-c(theta))], dim = -1)

    exp = torch.stack((r1, r2, r3), dim = 1)
    idx_0 = torch.abs(theta) < 1e-6
    exp[idx_0] = torch.eye(3).to(device).repeat(mat_.shape[0], 1, 1)[idx_0]

    return homo_from_mat_trans(t = mat_[:, 3:], r = exp.view(-1, 9))




# r1 = torch.tensor([[-7.0711e-01,  7.0710e-01, -7.8603e-06,
#                      7.0710e-01,  7.0711e-01, -1.2909e-06,
#                      4.6453e-06, -6.4708e-06, -1.0000e+00]]).repeat(2,1)
# t1 = torch.tensor([[-4.9190e-01,  1.3330e-01,  4.8790e-01]]).repeat(2,1)


# r2 = torch.tensor([[-0.8705,  0.1751,  0.4600,
#                     -0.0447,  0.9026, -0.4281,
#                     -0.4901, -0.3932, -0.7779]])
# t2 = torch.tensor([[-0.1,  0.1348,  0.3480]])


# r2 = torch.tensor([[ 1.0000,  0.0000,  0.0000,
#                     -0.0000, -0.5000,  0.8660,
#                      0.0000, -0.8660, -0.5000]]).repeat(2,1)
# t2 = torch.tensor([[-0.0,  0.01,  0.0]]).repeat(2,1)


# h1 = homo_from_mat_trans(t1, r1)
# h2 = homo_from_mat_trans(t2, r2)

# print("H1: ", torch.round(h1, decimals = 4))
# print("H2: ", torch.round(h2, decimals = 4))

# h_diff = mat_diff(h1, h2)

# print("DIFF: ", torch.round(h_diff, decimals = 4))

# l_ = log_mat(mat = h_diff)
# b = exp_mat(mat_ = l_)


# print("DIFF: ", b.round(decimals = 4))
# print("PRE Logaritmico: ", l_.round(decimals = 4))

# b = mat_mul(h1, b)

# print("Logaritmico: ", l_.round(decimals = 4))
# print("Exponencial: ", b.round(decimals = 4))
