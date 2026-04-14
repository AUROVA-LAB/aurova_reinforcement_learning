import torch
from .lab_utils import matrix_from_quat, quat_from_matrix


# ======== HOMOGENEOUS TRANSFORMATION MATRIX =======================================================================

def norm_mat(x: torch.Tensor):
    """
    Normalization of a homogeneous transformation matrix SE(3) -> not implemented yet

    In: 
        - x: any torch tensor
    Out:
        - res: the same tensor
    """

    return x


def convert_homo_to_Lab(x: torch.Tensor):
    """
    Convert a tensor of homogeneous transformation matrices x of the shape [*, 16] to the format of translation+quaternion

    In:
        - x: [B, 16]: tensor with homogeneous transformation matrices SE(3)

    Out:
        - res: [B, 7]: tensor with the corresponding translation+quaternion (x,y,z, w,x,y,z)
    """

    assert x.shape[-1] == 16

    # Convert the rotation matrices
    x = x.view(-1, 4, 4)
    q = quat_from_matrix(matrix = x[:, :-1, :-1])

    # Concatenates the result
    return torch.cat((x[:, :-1, -1], q), dim = -1)


def homo_from_mat_trans_LAB(t: torch.Tensor, r: torch.tensor):
    """
    Convert tensors in the format of translation+quaternion to a homogeneous transformation matrix

    In: 
        - t: [B, 3]: tensor with translations (x,y,z)
        - r: [B, 4]: tensor with quaternions (w,x,y,z)

    Out:
        - res: [B, 16]: tensor with the corresponding homogeneous transformation matrix
    """
    assert t.shape[-1] == 3
    assert r.shape[-1] == 4

    device = t.device

    # Convert quaternion to rotation matrix
    r = matrix_from_quat(r)

    # Concatenate rotation and translation
    cat1 = torch.cat((r.view(-1, 3,3), t.unsqueeze(-1)), dim = -1).view(-1, 12)
    
    # Declare scaling row
    row4 = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(t.shape[0], 1).to(device)

    # Concatenates scaling row
    return torch.cat((cat1, row4), dim = -1).view(-1, 16)


def mat_diff(m1: torch.Tensor, m2: torch.Tensor):
    """
    Difference of two homogeneous transformation matrices m1 and m2

    In:
        - m1: [B, 16]: tensor with a set of homogeneous transformation matrices SE(3)
        - m2: [B, 16]: tensor with another set of homogeneous transformation matrices SE(3)
    Out:
        - res: [B, 16]: tensor with the difference between both tensors
    """

    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    T1 = m1.view(-1, 4,4)
    T2 = m2.view(-1, 4,4)

    # Extract rotation and translation
    R1 = T1[:, :3, :3]  
    t1 = T1[:, :3, 3] 

    R2 = T2[:, :3, :3]
    t2 = T2[:, :3, 3]

    # Inverse of T1
    R1_inv = R1.transpose(1, 2)  
    t1_inv = -(R1_inv @ t1.unsqueeze(-1)).squeeze(-1)

    # Relative transform
    R_diff = R1_inv @ R2  
    t_diff = (R1_inv @ (t2 - t1).unsqueeze(-1)).squeeze(-1)

    # Assemble homogeneous transformation matrices
    T_diff = torch.eye(4, dtype=T1.dtype, device=T1.device).unsqueeze(0).repeat(T1.shape[0], 1, 1)
    T_diff[:, :3, :3] = R_diff
    T_diff[:, :3, 3] = t_diff

    return T_diff.view(-1, 16)


def mat_mul(m1: torch.Tensor, m2: torch.Tensor):
    """
    Multiplication of two homogeneous transformation matrices m1 and m2

    In:
        - m1: [B, 16]: tensor with a set of homogeneous transformation matrices SE(3)
        - m2: [B, 16]: tensor with another set of homogeneous transformation matrices SE(3)
    Out:
        - res: [B, 16]: tensor with the multiplication of both tensors
    """

    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    return torch.matmul(m1.view(-1, 4, 4), m2.view(-1, 4, 4)).view(-1, 16)
