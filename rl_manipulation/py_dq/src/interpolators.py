import torch
from .dq import *

# === INTERPOLATION METHODS =======================================================================

def ScLERP(p1: torch.Tensor, p2: torch.Tensor, exp, log, diff, mul, norm, step: int = 0.1):
    """
    Screw Linear Interpolation  between two poses p1 and p2 specifying the step, the group and mapping operations.
    """
    assert p1.shape[-1] == 8
    assert p2.shape[-1] == 8
    assert torch.any(torch.logical_and(dq_is_norm(p1), dq_is_norm(p1)))
    
    device = p1.device

    # Time vector
    tau = torch.arange(0, 1 + step, step).to(device)

    # Relative pose or Difference between p1 and p2
    dq_diff = diff(p1 = p1, p2 = p2)

    # Logarithm mapping of the relative difference
    log_diff = log(dq = dq_diff)

    # Scale the difference according to tau and map it to the group: exp(diff*tau)
    #      Obtains a linear interpolation from [0,0,0,0,0,0] to log(dq_diff)
    dq_diff_tau = exp(dq_ = (tau*log_diff.unsqueeze(-1)).transpose(1,2).reshape(-1, 6))

    # The sclaed differences are put into the original reference frame
    dq_tau = mul(p1 = p1.repeat_interleave(tau.shape[0], dim = 0), p2 = dq_diff_tau)

    return norm(dq = dq_tau).view(p1.shape[0], -1, 8)


