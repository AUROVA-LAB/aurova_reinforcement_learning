import torch
from .matrix import *


def hat(vec):
    """
    Converts a vector [*, 3] to its skew-symmetric matric [*,3,3].
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
    Converts a  skew-symmetric matric mat of the shape [*,3,3] to a vector [*, 3]. 
    """
    return torch.stack([mat[:,2,1], mat[:,0,2], mat[:,1,0]], dim=-1)

def homo_from_rt(R, t):
    """
    Builds a Homogeneous Transformation Matrix from a rotation matrix R of the shape [*,3,3] 
        and a translation vector t of the shape [*, 3]
    """
    B = R.shape[0]
    H = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)
    H[:,:3,:3] = R
    H[:,:3, 3] = t

    return H



def log_se3(T: torch.Tensor):
    """
    Logarithm of a Homogeneous Transformation Matrix.
    """
    B = T.shape[0]

    T = T.view(-1, 4, 4)

    R = T[:,:3,:3]
    t = T[:,:3, 3]

    # trace and angle
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((tr - 1)/2, -1, 1)
    theta = torch.acos(cos_theta)

    sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, 0))
    theta = torch.atan2(sin_theta, cos_theta)


    # Near zero cases
    is_small = theta < 1e-6

    # Rotation
    K = 0.5 * (R - R.transpose(-1,-2))
    phi = vee(K)  # = (sinθ/θ)*ω
    scale = torch.ones_like(theta)
    scale[~is_small] = theta[~is_small] / torch.sin(theta[~is_small])
    phi = phi * scale.unsqueeze(-1)


    # Build skew
    A = hat(phi)
    A2 = A @ A
    I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)

    # J^{-1}
    J_inv = I - 0.5*A
    alpha = torch.zeros_like(theta)
    mask = ~is_small
    alpha[mask] = 1/theta[mask]**2 - (1+torch.cos(theta[mask]))/(2*theta[mask]*torch.sin(theta[mask]))
    J_inv = J_inv + alpha.view(-1,1,1)*A2
    
    # Near zero case
    J_inv[is_small] = (I - 0.5*A + (1/12.0)*A2)[is_small]


    rho = torch.bmm(J_inv, t.unsqueeze(-1)).squeeze(-1)

    xi = torch.cat([phi, rho], dim=-1)
    return torch.round(xi, decimals =3)




def exp_se3(xi: torch.Tensor):
    """
    Exponential map of a Homogeneous Transformation Matrix.
    """
    B = xi.shape[0]
    phi = xi[:,:3]
    rho = xi[:,3:]

    theta = torch.linalg.norm(phi, dim=-1)
    is_small = theta < 1e-6

    A = hat(phi)
    A2 = A @ A
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)

    # Rotation
    R = I + A + 0.5*A2
    mask = ~is_small
    if mask.any():
        th = theta[mask]
        R[mask] = I[mask] \
                  + (torch.sin(th)/th).view(-1,1,1)*A[mask] \
                  + ((1-torch.cos(th))/th**2).view(-1,1,1)*A2[mask]

    # J
    J = I + 0.5*A + (1/6.0)*A2
    if mask.any():
        th = theta[mask]
        J[mask] = I[mask] \
                  + ((1-torch.cos(th))/th**2).view(-1,1,1)*A[mask] \
                  + ((th-torch.sin(th))/th**3).view(-1,1,1)*A2[mask]

    t = torch.bmm(J, rho.unsqueeze(-1)).squeeze(-1)

    return torch.round(homo_from_rt(R, t).view(B, -1), decimals =3)


