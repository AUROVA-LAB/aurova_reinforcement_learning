import torch
from .matrix import *
import torch.nn.functional as F
import math



def hat(vec):
    """[..., 3] -> [..., 3, 3]."""
    x, y, z = vec.unbind(-1)
    o = torch.zeros_like(x)

    return torch.stack(
        (o, -z, y, z, o, -x, -y, x, o),
        dim=-1,
    ).reshape(vec.shape[:-1] + (3, 3))


def vee(mat):
    """[..., 3, 3] -> [..., 3], para matrices antisimétricas."""
    return torch.stack(
        (
            mat[..., 2, 1],
            mat[..., 0, 2],
            mat[..., 1, 0],
        ),
        dim=-1,
    )


def homo_from_rt(R, t):
    """
    R: [..., 3, 3]
    t: [..., 3]
    Salida: [..., 4, 4]
    """
    upper = torch.cat((R, t.unsqueeze(-1)), dim=-1)

    bottom = R.new_tensor(
        [0.0, 0.0, 0.0, 1.0]
    ).expand(R.shape[:-2] + (1, 4))

    return torch.cat((upper, bottom), dim=-2)


def _q_to_rotvec(q):
    """
    Cuaternión -> vector de rotación principal phi.

    Selecciona qw >= 0, por lo que ||phi|| <= pi.
    En pi existe la ambigüedad inevitable del eje.
    """
    q = q / torch.linalg.vector_norm(
        q, dim=-1, keepdim=True
    )

    q = torch.where(q[..., :1] < 0, -q, q)

    w = q[..., 0]
    v = q[..., 1:]
    s = torch.linalg.vector_norm(v, dim=-1)

    small = s < 1e-3
    scale = torch.empty_like(s)

    # 2*asin(s)/s alrededor de s = 0.
    s2 = s[small] ** 2
    scale[small] = 2 + s2 / 3 + 3 * s2**2 / 20

    scale[~small] = (
        2 * torch.atan2(s[~small], w[~small])
        / s[~small]
    )

    return v * scale.unsqueeze(-1)


def _so3_log(R):
    """
    Logaritmo de SO(3).

    Recupera primero el cuaternión mediante su componente
    mejor condicionada, evitando dividir por sin(theta)
    cerca de theta = pi.
    """
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]

    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]

    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    # [4*qw², 4*qx², 4*qy², 4*qz²].
    four_sq = torch.stack(
        (
            1 + r00 + r11 + r22,
            1 + r00 - r11 - r22,
            1 - r00 + r11 - r22,
            1 - r00 - r11 + r22,
        ),
        dim=-1,
    )

    candidates = torch.stack(
        (
            torch.stack(
                (
                    four_sq[..., 0],
                    r21 - r12,
                    r02 - r20,
                    r10 - r01,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    r21 - r12,
                    four_sq[..., 1],
                    r10 + r01,
                    r02 + r20,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    r02 - r20,
                    r10 + r01,
                    four_sq[..., 2],
                    r12 + r21,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    r10 - r01,
                    r02 + r20,
                    r12 + r21,
                    four_sq[..., 3],
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )

    idx = four_sq.argmax(dim=-1)

    row = torch.gather(
        candidates,
        dim=-2,
        index=idx[..., None, None].expand(
            idx.shape + (1, 4)
        ),
    ).squeeze(-2)

    best = four_sq.gather(
        dim=-1,
        index=idx.unsqueeze(-1),
    )

    q = row / (2 * torch.sqrt(best))

    return _q_to_rotvec(q)


def _so3_exp_and_jacobian(phi):
    """
    Devuelve:
        R = Exp_SO3(phi)
        J = Jacobiano izquierdo J_l(phi)
    """
    theta = torch.linalg.vector_norm(phi, dim=-1)

    A = hat(phi)
    A2 = A @ A

    I = torch.eye(
        3,
        device=phi.device,
        dtype=phi.dtype,
    ).expand(A.shape)

    # torch.sinc(x) = sin(pi*x)/(pi*x).
    a = torch.sinc(theta / math.pi)

    b = (
        0.5
        * torch.sinc(theta / (2 * math.pi)) ** 2
    )

    # c = (theta - sin(theta))/theta³.
    # Serie para evitar cancelación numérica cerca de cero.
    small = theta < 0.05
    c = torch.empty_like(theta)

    th2 = theta[small] ** 2

    c[small] = (
        1 / 6
        - th2 / 120
        + th2**2 / 5040
        - th2**3 / 362880
    )

    th = theta[~small]
    c[~small] = (th - torch.sin(th)) / th**3

    R = (
        I
        + a[..., None, None] * A
        + b[..., None, None] * A2
    )

    J = (
        I
        + b[..., None, None] * A
        + c[..., None, None] * A2
    )

    return R, J


def log_se3(T: torch.Tensor, so3=False):
    """
    Entrada:
        T: [..., 16] o [..., 4, 4].

    Salida:
        so3=False: [phi, rho], rho = J_l(phi)^(-1) @ t.
        so3=True:  [phi, t].

    so3=True es una parametrización separada de la pose,
    no el logaritmo completo de SE(3).
    """
    if T.shape[-1] == 16:
        T = T.reshape(T.shape[:-1] + (4, 4))

    elif T.shape[-2:] != (4, 4):
        raise ValueError(
            "T debe tener forma [...,16] o [...,4,4]"
        )

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    phi = _so3_log(R)

    if so3:
        return torch.cat((phi, t), dim=-1)

    _, J = _so3_exp_and_jacobian(phi)

    # Resuelve J @ rho = t sin invertir J explícitamente.
    rho = torch.linalg.solve(
        J,
        t.unsqueeze(-1),
    ).squeeze(-1)

    return torch.cat((phi, rho), dim=-1)


def exp_se3(xi: torch.Tensor, so3=False):
    """
    Entrada:
        xi: [..., 6].

        so3=False: xi = [phi, rho].
        so3=True:  xi = [phi, t].

    Salida:
        Transformación aplanada [..., 16].
    """
    if xi.shape[-1] != 6:
        raise ValueError(
            "xi debe tener forma [...,6]"
        )

    phi = xi[..., :3]
    rho = xi[..., 3:]

    R, J = _so3_exp_and_jacobian(phi)

    if so3:
        t = rho
    else:
        t = (J @ rho.unsqueeze(-1)).squeeze(-1)

    T = homo_from_rt(R, t)

    return T.reshape(xi.shape[:-1] + (16,))



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

