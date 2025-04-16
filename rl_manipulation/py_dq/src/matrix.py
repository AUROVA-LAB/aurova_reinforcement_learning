import torch
from omni.isaac.lab.utils.math import matrix_from_quat, quat_from_matrix


# ======== MATRIX =======================================================================

def norm_mat(x: torch.Tensor):
    return x


def homo_from_mat_trans_LAB(t: torch.Tensor, r: torch.tensor):
    assert t.shape[-1] == 3
    assert r.shape[-1] == 4

    device = t.device

    r = matrix_from_quat(r)

    cat1 = torch.cat((r.view(-1, 3,3), t.unsqueeze(-1)), dim = -1).view(-1, 12)
    row4 = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(t.shape[0], 1).to(device)

    return torch.cat((cat1, row4), dim = -1).view(-1, 16)



def homo_from_mat_trans(t: torch.Tensor, r: torch.tensor):
    assert t.shape[-1] == 3
    assert r.shape[-1] == 9

    device = t.device

    cat1 = torch.cat((r.view(-1, 3,3), t.unsqueeze(-1)), dim = -1).view(-1, 12)
    row4 = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(t.shape[0], 1).to(device)

    return torch.cat((cat1, row4), dim = -1).view(-1, 16)


def mat_diff(m1: torch.Tensor, m2: torch.Tensor):
    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    return torch.matmul(torch.inverse(m1.view(-1, 4, 4)), m2.view(-1, 4, 4)).view(-1, 16)

def mat_mul(m1: torch.Tensor, m2: torch.Tensor):
    assert m1.shape[-1] == m2.shape[-1]
    assert m1.shape[-1] == 16

    return torch.matmul(m1.view(-1, 4, 4), m2.view(-1, 4, 4)).view(-1, 16)
