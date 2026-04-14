import torch
from .matrix import *
import torch.nn.functional as F


def hat(vec):
    """
    Converts a vector [*, 3] to its skew-symmetric matric [*,3,3].

    In:
        - vec: [B, 3]: tensor with a set of skew angles (x,y,z)

    Out:
        - res: [B, 3, 3]: tensor with the corresponding skew-symmetric matrices
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

    In:
        - mat: [B, 3, 3]: tensor with a set of skew-symmetric matrices

    Out:
        - res: [B, 3]: tensor with the corresponding skew angles (x,y,z)
    """

    return torch.stack([mat[:,2,1], mat[:,0,2], mat[:,1,0]], dim=-1)


def homo_from_rt(R, t):
    """
    Builds a Homogeneous Transformation Matrix from a rotation matrix R of the shape [*,3,3] 
        and a translation vector t of the shape [*, 3]
    
    In: 
        - R: [B, 3, 3]: tensor with a set of rotation matrices
        - t: [B, 3]: tensor with a set of translation vectors

    Out:
        - res: [B, 4, 4]: tensor with the corresponding homogenous transformation matrices
    """

    B = R.shape[0]
    H = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)
    H[:,:3,:3] = R
    H[:,:3, 3] = t

    return H


def log_se3(T: torch.Tensor, so3 = False):
    """
    Logarithm of a Homogeneous Transformation Matrix.

    In:
        - T: [B, 16]: tensor with a set of homogenous transformation matrices
        - so3: [bool]: wether use the SO(3) or the SE(3) mapping functions

    Out:
        - res: [B, 6]: tensor with the corresponding twist coordinates
    """

    B = T.shape[0]

    T = T.view(-1, 4, 4)

    # Separate rotation and translation
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

    # --- Map Rotation ---
    K = 0.5 * (R - R.transpose(-1,-2))
    phi = vee(K)  # = (sinθ/θ)*ω
    scale = torch.ones_like(theta)
    scale[~is_small] = theta[~is_small] / torch.sin(theta[~is_small])
    phi = phi * scale.unsqueeze(-1)

    # "so3=True" -> only map rotation
    if so3:
        xi = torch.cat([phi, t], dim=-1)
    
    # "so3=False" -> map rotation and translatoin
    else:
        # --- Map Translation ---
        # Build skew
        A = hat(phi)
        A2 = A @ A
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B,1,1)

        # Inverse Jacobian J^{-1}
        J_inv = I - 0.5*A
        alpha = torch.zeros_like(theta)
        mask = ~is_small
        alpha[mask] = 1/theta[mask]**2 - (1+torch.cos(theta[mask]))/(2*theta[mask]*torch.sin(theta[mask]))
        J_inv = J_inv + alpha.view(-1,1,1)*A2
        
        # Near zero case
        J_inv[is_small] = (I - 0.5*A + (1/12.0)*A2)[is_small]

        # Map translation
        rho = torch.bmm(J_inv, t.unsqueeze(-1)).squeeze(-1)
        xi = torch.cat([phi, rho], dim=-1)

    # Round the result
    return torch.round(xi, decimals =3)


def exp_se3(xi: torch.Tensor, so3 = False):
    """
    Exponential map of a Homogeneous Transformation Matrix.

    In:
        - xi: [B, 6]: tensor with a set of twist elements from SE(3) or SO(3)
        - so3: [bool]: wether use the SO(3) or the SE(3) mapping functions

    Out:
        - res: [B, 4, 4]: tensor with the corresponding homogeneous transformation matrices
    """

    B = xi.shape[0]
    
    # Separate rotation and translation
    phi = xi[:,:3]
    rho = xi[:,3:]

    # Obtain rotation angle
    theta = torch.linalg.norm(phi, dim=-1)
    
    # Near zero flag
    is_small = theta < 1e-6

    A = hat(phi)
    A2 = A @ A
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)

    # --- Map Rotation ---
    R = I + A + 0.5*A2
    
    # "so3=True" -> only map rotation
    if so3:
        # Rounds the result
        return torch.round(homo_from_rt(R, rho).view(B, -1), decimals =3)
    
    # "so3=False" -> map rotation and translation
    else:
        # --- Map Translation ---
        # Distinguish if there is near zero cases
        mask = ~is_small
        if mask.any():
            th = theta[mask]
            R[mask] = I[mask] \
                    + (torch.sin(th)/th).view(-1,1,1)*A[mask] \
                    + ((1-torch.cos(th))/th**2).view(-1,1,1)*A2[mask]

        # Jacobian
        J = I + 0.5*A + (1/6.0)*A2
        if mask.any():
            th = theta[mask]
            J[mask] = I[mask] \
                    + ((1-torch.cos(th))/th**2).view(-1,1,1)*A[mask] \
                    + ((th-torch.sin(th))/th**3).view(-1,1,1)*A2[mask]

        # Maps the translation
        t = torch.bmm(J, rho.unsqueeze(-1)).squeeze(-1)

        # Rounds the result
        return torch.round(homo_from_rt(R, t).view(B, -1), decimals =3)
    

def log_gram(R: torch.Tensor):
    """
    Conversion to a continous version of a rotation matrices SO(3) -> On the Continuity of Rotation Representations in Neural Networks

    In:
        - R: [B, 9]: tensor with a set of rotation matrices SO(3)

    Out: 
        - res: [B, 6]: tensor of continuous rotation matrices
    """

    # The last colum is removed
    return R[:, :, :-1].transpose(-1, -2)


def exp_gram(R_: torch.Tensor, eps = 1e-08):
    """
    Conversion to SO(3) matrices from its continous version -> On the Continuity of Rotation Representations in Neural Networks

    In: 
        R_: (B, 6): tensor with a set of continous rotation matrices
    Out:
        res: (B, 9): tensor of the corresponding rotation matrices SO(3)
    """

    a1 = R_[:, 0:3]             # (B, 3)
    a2 = R_[:, 3:6]             # (B, 3)

    # Normalize first vector
    b1 = F.normalize(a1, dim=1, eps=eps)

    # Gram–Schmidt: remove component of a2 along b1
    dot = torch.sum(b1 * a2, dim=1, keepdim=True)   # (B, 1)
    a2_orth = a2 - dot * b1

    # Normalize second vector
    b2 = F.normalize(a2_orth, dim=1, eps=eps)

    # Third basis vector via cross product
    b3 = torch.cross(b1, b2, dim=1)

    # Stack into rotation matrix
    R = torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)

    return R


def log_gram_se3(T: torch.Tensor):
    """
    Conversion to a continous version of a homogeneous transformation matrices -> On the Continuity of Rotation Representations in Neural Networks

    In:
        - T: [B, 16]: tensor with a set of homogeneous transformation matrices SE(3)

    Out: 
        - res: [B, 12]: tensor with the corresponding continuous homogeneous transformation matrices
    """

    B = T.shape[0]

    T = T.view(-1, 4, 4)

    # Separate rotation and translation
    R = T[:,:3,:3]
    t = T[:,:3, 3]

    # Make rotation continuous
    R_ = log_gram(R = R).reshape(B, -1)

    # Concatenate both terms
    return torch.cat((R_, t), dim = -1)


def exp_gram_se3(T_: torch.Tensor):
    """
    Conversion from a continous version of a homogeneous transformation matrices to SE(3) -> On the Continuity of Rotation Representations in Neural Networks

    In:
        - T_: [B, 12]: tensor with a set of a continuous homogeneous transformation matrices 

    Out: 
        - res: [B, 4, 4]: tensor with the corresponding homogeneous transformation matrices SE(3)
    """
    
    B = T_.shape[0]

    # Separate rotation and translation
    R_ = T_[:, :6]
    t = T_[:, 6:]

    # Exponential of the rotation part
    R = exp_gram(R_ = R_)

    # Build the homogeneous transformation matrix and round it
    return torch.round(homo_from_rt(R, t).view(B, -1), decimals =5)

