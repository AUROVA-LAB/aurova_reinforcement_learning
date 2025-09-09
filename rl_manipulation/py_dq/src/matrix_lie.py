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


# =========================
# Utilidades
# =========================

def hat(vec):
    """
    Convierte un vector (B,3) en su matriz skew-symmetric (B,3,3).
    """
    B = vec.shape[0]
    x, y, z = vec[:,0], vec[:,1], vec[:,2]
    O = torch.zeros(B, device=vec.device, dtype=vec.dtype)
    K = torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1),
    ], dim=1)
    return K

def vee(mat):
    """
    Convierte una matriz skew-symmetric (B,3,3) en su vector (B,3).
    """
    return torch.stack([mat[:,2,1], mat[:,0,2], mat[:,1,0]], dim=-1)

def homo_from_rt(R, t):
    """
    Construye una matriz homogénea a partir de R (B,3,3) y t (B,3).
    """
    B = R.shape[0]
    H = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)
    H[:,:3,:3] = R
    H[:,:3, 3] = t
    return H

# =========================
# Logaritmo de SE(3)
# =========================

def log_se3(T: torch.Tensor):
    """
    Logaritmo de una matriz homogénea SE(3).
    T: (B,4,4)
    Devuelve: xi (B,6) donde xi = [phi, rho]
    """
    B = T.shape[0]

    T = T.view(-1, 4, 4)

    R = T[:,:3,:3]
    t = T[:,:3, 3]

    # Traza y ángulo
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((tr - 1)/2, -1, 1)
    theta = torch.acos(cos_theta)

    # Manejo numérico estable con atan2
    sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, 0))
    theta = torch.atan2(sin_theta, cos_theta)


    # Casos pequeños
    is_small = theta < 1e-6

    # Extraer rotación
    K = 0.5 * (R - R.transpose(-1,-2))
    phi = vee(K)  # = (sinθ/θ)*ω
    scale = torch.ones_like(theta)
    scale[~is_small] = theta[~is_small] / torch.sin(theta[~is_small])
    phi = phi * scale.unsqueeze(-1)


    # Construir skew
    A = hat(phi)
    A2 = A @ A
    I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)

    # J^{-1}
    J_inv = I - 0.5*A
    alpha = torch.zeros_like(theta)
    mask = ~is_small
    alpha[mask] = 1/theta[mask]**2 - (1+torch.cos(theta[mask]))/(2*theta[mask]*torch.sin(theta[mask]))
    J_inv = J_inv + alpha.view(-1,1,1)*A2
    # Serie para caso pequeño
    J_inv[is_small] = (I - 0.5*A + (1/12.0)*A2)[is_small]


    rho = torch.bmm(J_inv, t.unsqueeze(-1)).squeeze(-1)

    xi = torch.cat([phi, rho], dim=-1)
    return xi

# =========================
# Exponencial de se(3)
# =========================

def exp_se3(xi: torch.Tensor):
    """
    Exponencial de un twist en se(3).
    xi: (B,6) = [phi, rho]
    Devuelve: T (B,4,4)
    """
    B = xi.shape[0]
    phi = xi[:,:3]
    rho = xi[:,3:]

    theta = torch.linalg.norm(phi, dim=-1)
    is_small = theta < 1e-6

    A = hat(phi)
    A2 = A @ A
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)

    # Rotación
    R = I + A + 0.5*A2
    mask = ~is_small
    if mask.any():
        th = theta[mask]
        R[mask] = I[mask] \
                  + (torch.sin(th)/th).view(-1,1,1)*A[mask] \
                  + ((1-torch.cos(th))/th**2).view(-1,1,1)*A2[mask]

    # Jacobiano
    J = I + 0.5*A + (1/6.0)*A2
    if mask.any():
        th = theta[mask]
        J[mask] = I[mask] \
                  + ((1-torch.cos(th))/th**2).view(-1,1,1)*A[mask] \
                  + ((th-torch.sin(th))/th**3).view(-1,1,1)*A2[mask]

    t = torch.bmm(J, rho.unsqueeze(-1)).squeeze(-1)

    return homo_from_rt(R, t).view(B, -1)



# def log_mat_pre(mat: torch.tensor):
#     assert mat.shape[-1] == 16

#     device = mat.device

#     mat = mat.view(-1, 4, 4)
#     t = mat[:, :-1, -1]
#     mat = mat[:, :-1, :-1]
    
#     theta = torch.acos(torch.round(0.5 * (torch.vmap(torch.trace)(mat) - 1), decimals = 4))
#     trace = 0.5 * (torch.vmap(torch.trace)(mat) - 1)

#     log = (theta / (2*torch.sin(theta)) * torch.stack([mat[:, 2,1] - mat[:, 1,2], 
#                                                        mat[:, 0,2] - mat[:, 2,0],
#                                                        mat[:, 1,0] - mat[:, 0,1]])).transpose(0,1)

#     idx_0 = torch.isclose(theta, torch.tensor(0.0).float())
#     idx_pi = torch.isclose(theta, torch.tensor(math.pi))
#     idx_mpi = torch.isclose(theta, torch.tensor(-math.pi))


#     log[idx_0] = torch.zeros_like(log)[idx_0]

#     S = mat + mat.transpose(-1, -2) + (1 - torch.vmap(torch.trace)(mat))[:, None, None]*torch.eye(3).to(device)
#     S = torch.stack((S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]), dim = -1)
#     n = S / (3 - torch.vmap(torch.trace)(mat))[:, None]

#     idx_3 = torch.isclose(torch.vmap(torch.trace)(mat), 3*torch.ones((mat.shape[0])).to(device))

#     R = torch.stack((mat[:, 0, 0], mat[:, 1, 1], mat[:, 2, 2]), dim = -1)

#     log[idx_pi] = (n)[idx_pi]
#     log[idx_mpi] = (n)[idx_mpi]
#     log[idx_3] = torch.zeros_like(log)[idx_3]

#     return torch.cat((log, t), dim = -1)

# def log_mat(mat: torch.tensor):
#     assert mat.shape[-1] == 16

#     device = mat.device

#     mat = mat.view(-1, 4, 4)
#     R = mat[:, :-1, :-1]
#     t = mat[:, :-1, -1]

#     eps = 1e-6

#     I = torch.eye(3).repeat(mat.shape[0], 1, 1)


#     # ---- Compute theta ----
#     # Trace
#     c = torch.clamp((torch.vmap(torch.trace)(R)  - 1)/2, -1, 1)
    
#     # Base zero tensor
#     # theta = torch.zeros(mat.shape[0]).to(device)

#     # # Indexes for zeros
    
#     # The non-zero are assigned the trace cosinus
#     theta = torch.acos(c)
#     is_zero = torch.isclose(theta, torch.zeros_like(theta), rtol = eps)


#     # ---- Compute exp(R) ----
#     # Skew of R
#     A_skew = 0.5 * (R - R.transpose(-1, -2))
    
#     # Skew indexes 
#     idx_skew = torch.tensor([7, 2, 3])
    
#     # Select skew elements
#     A_skew_f = A_skew.view(A_skew.shape[0], -1)
#     A = A_skew_f[:, idx_skew]
 
#     # The non-zero are assigned the extended calculus
#     A[~is_zero] *= (theta[~is_zero] / torch.sin(theta[~is_zero])).unsqueeze(-1)
#     A_skew = hat(A)


#     # ---- Inverse Left Jacobian ----
#     J_inv = I - 0.5 * A_skew
#     J_inv[is_zero]  = (I - 0.5 * A_skew + 1 / 12 * (A_skew@A_skew))[is_zero]
#     J_inv[~is_zero] += (1 / theta[~is_zero]**2 - (1 + torch.cos(theta[~is_zero])) / (2 * theta[~is_zero] * torch.sin(theta[~is_zero]))).unsqueeze(-1).unsqueeze(-1) * (A_skew@A_skew)[~is_zero]


#     # ---- Coupled lineal part ----
#     v = torch.bmm(J_inv, t.unsqueeze(-1)).squeeze(-1)

#     return torch.cat((A, v), dim = -1)


# def exp_mat_pre(mat_: torch.tensor):
#     assert mat_.shape[-1] == 6

#     device = mat_.device

#     theta = torch.norm(mat_[:, :3], dim = -1)

#     n = mat_[:, :3] / theta.unsqueeze(-1)
    
#     c = torch.cos
#     s  = torch.sin

#     # exp = torch.sin(theta) / theta *  mat_[:, :3] + (1 - torch.cos(theta)) / theta**2 * mat_[:, :3]**2

#     r1 = torch.stack([c(theta) + n[:, 0]**2 * (1-c(theta)),                     
#                        n[:, 0] * n[:, 1] * (1 - c(theta)) - n[:, 2]*s(theta),    
#                        n[:, 0] * n[:, 2] * (1 - c(theta)) + n[:, 1]*s(theta)], dim = -1)
#     r2 = torch.stack([n[:, 0] * n[:, 1] * (1 - c(theta)) + n[:, 2]*s(theta),    
#                       c(theta) + n[:, 1]**2 * (1-c(theta)),                     
#                       n[:, 1] * n[:, 2] * (1 - c(theta)) - n[:, 0]*s(theta)], dim = -1)
#     r3 = torch.stack([n[:, 0] * n[:, 2] * (1 - c(theta)) - n[:, 1]*s(theta),    
#                       n[:, 1] * n[:, 2] * (1 - c(theta)) + n[:, 0]*s(theta),    
#                       c(theta) + n[:, 2]**2 * (1-c(theta))], dim = -1)

#     exp = torch.stack((r1, r2, r3), dim = 1)
#     idx_0 = torch.abs(theta) < 1e-6
#     exp[idx_0] = torch.eye(3).to(device).repeat(mat_.shape[0], 1, 1)[idx_0]


#     return homo_from_mat_trans(t = mat_[:, 3:], r = exp.view(-1, 9))

# def exp_mat(mat_: torch.tensor, kwargs = None):
#     assert mat_.shape[-1] == 6

#     device = mat_.device
    
#     A_skew = kwargs["A_skew"]
#     theta = kwargs["theta"]
#     is_zero = kwargs["is_zero"]

#     # ---- Rotation matrix ----


#     '''
#     Poner EYE en vez del 1, se la identidad'''



#     R = 1 + A_skew + 0.5 * (A_skew@A_skew)
#     R[~is_zero] = 1 + (torch.sin(theta[~is_zero]) / theta[~is_zero]).unsqueeze(-1).unsqueeze(-1)*A_skew[~is_zero] + \
#                 ((1 - torch.cos(theta[~is_zero])) / theta[~is_zero]**2).unsqueeze(-1).unsqueeze(-1) * (A_skew@A_skew)[~is_zero]
    

#     # ----Left Jacobian ----
#     J = 1 + 0.5 * A_skew + 1/6 * (A_skew@A_skew)
#     J[~is_zero] = 1 + ((1 - torch.cos(theta[~is_zero])) / (theta[~is_zero]**2)).unsqueeze(-1).unsqueeze(-1) * A_skew[~is_zero] + \
#                 ((theta[~is_zero] - torch.sin(theta[~is_zero])) / (theta[~is_zero]**3)).unsqueeze(-1).unsqueeze(-1) * (A_skew@A_skew)[~is_zero]
    
#     t = torch.bmm(J, mat_[:, 3:].unsqueeze(-1)).squeeze(-1)

#     return homo_from_mat_trans(t = t, r = R.view(-1, 9))



# r1 = torch.tensor([[-7.0711e-01,  7.0710e-01, -7.8603e-06,
#                      7.0710e-01,  7.0711e-01, -1.2909e-06,
#                      4.6453e-06, -6.4708e-06, -1.0000e+00]]).repeat(4,1)
# t1 = torch.tensor([[-4.9190e-01,  1.3330e-01,  4.8790e-01]]).repeat(4,1)


# r2 = torch.tensor([[ 1.0000,  0.0000,  0.0000,
#                     -0.0000, -0.5000,  0.8660,
#                      0.0000, -0.8660, -0.5000]])
# t2 = torch.tensor([[-0.1,  0.1348,  0.3480]])


# r2 = torch.tensor([[ 1.0000,  0.0000,  0.0000,
#                     -0.0000, -0.5000,  0.8660,
#                      0.0000, -0.8660, -0.5000]]).repeat(4,1)
# t2 = torch.tensor([[-0.0,  0.01,  0.0]]).repeat(4,1)


# h1 = homo_from_mat_trans(t1, r1)
# h2 = homo_from_mat_trans(t2, r2)

# print("H1: ", torch.round(h1, decimals = 4).view(-1, 4, 4))
# print("H2: ", torch.round(h2, decimals = 4).view(-1, 4, 4))

# h_diff = mat_diff(h1, h2)
# h_diff[0] = h2[0]


# print("DIFF: ", torch.round(h_diff, decimals = 4).view(-1, 4, 4))

# l_= log_se3(T = torch.round(h_diff, decimals = 4).view(-1, 4, 4))
# b = exp_se3(xi = l_)


# print("DIFF: ", b.round(decimals = 4).view(-1, 4, 4))
# print("PRE Logaritmico: ", l_.round(decimals = 4))

# # b = mat_mul(h1, b)

# # print("Logaritmico: ", l_.round(decimals = 4))
# # print("Exponencial: ", b.round(decimals = 4))
