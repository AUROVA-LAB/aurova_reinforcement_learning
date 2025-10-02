import torch
from omni.isaac.lab.utils.math import matrix_from_quat, quat_from_matrix


# ======== HOMOGENEOUS TRANSFORMATION MATRIX =======================================================================

def norm_mat(x: torch.Tensor):
    return x

def convert_homo_to_Lab(x: torch.Tensor):
    """
    Convert an Homogeneous Transformation Matrix x of the shape [*, 16] to the format of IsaacLab (translation+quaternion).
    """
    assert x.shape[-1] == 16

    x = x.view(-1, 4, 4)
    q = quat_from_matrix(matrix = x[:, :-1, :-1])

    return torch.cat((x[:, :-1, -1], q), dim = -1)


def homo_from_mat_trans_LAB(t: torch.Tensor, r: torch.tensor):
    """
    Convert the format of IsaacLab (translation+quaternion) to a Homogeneous Transformation Matrix.
    """
    assert t.shape[-1] == 3
    assert r.shape[-1] == 4

    device = t.device

    r = matrix_from_quat(r)

    cat1 = torch.cat((r.view(-1, 3,3), t.unsqueeze(-1)), dim = -1).view(-1, 12)
    row4 = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(t.shape[0], 1).to(device)

    return torch.cat((cat1, row4), dim = -1).view(-1, 16)



def mat_diff(m1: torch.Tensor, m2: torch.Tensor):
    """
    Difference of two Homogeneous Transformation Matrices m1 and m2 of the shape [*, 16].
    """
    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    T1 = m1.view(-1, 4,4)
    T2 = m2.view(-1, 4,4)

    # Extract rotation and translation
    R1 = T1[:, :3, :3]  # (B,3,3)
    t1 = T1[:, :3, 3]   # (B,3)

    R2 = T2[:, :3, :3]
    t2 = T2[:, :3, 3]

    # Inverse of T1
    R1_inv = R1.transpose(1, 2)  # (B,3,3)
    t1_inv = -(R1_inv @ t1.unsqueeze(-1)).squeeze(-1)  # (B,3)

    # Relative transform
    R_diff = R1_inv @ R2  # (B,3,3)
    t_diff = (R1_inv @ (t2 - t1).unsqueeze(-1)).squeeze(-1)  # (B,3)

    # Assemble homogeneous matrices
    T_diff = torch.eye(4, dtype=T1.dtype, device=T1.device).unsqueeze(0).repeat(T1.shape[0], 1, 1)
    T_diff[:, :3, :3] = R_diff
    T_diff[:, :3, 3] = t_diff

    return T_diff.view(-1, 16)


def mat_mul(m1: torch.Tensor, m2: torch.Tensor):
    """
    Multiplication of two Homogeneous Transformation Matrices m1 and m2 of the shape [*, 16].
    """
    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    return torch.matmul(m1.view(-1, 4, 4), m2.view(-1, 4, 4)).view(-1, 16)
