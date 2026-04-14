import math
import torch

"""
Differentiable dual quaternion distance metric in PyTorch. 
Acknowledgements: 
- Function q_mul(: torch.Tensor): https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
- Other functions related to quaternions: re-implementations based on pip package "pyquaternion"
- Functions related to dual quaternions: re-implementations based on pip package "dual_quaternions"
"""

##################################### IMPORTANT ######################################
#### Code from                                                         ###############
#### https://gist.github.com/Flunzmas/d9485d9fee6244b544e7e75bdc0c352c ###############
######################################################################################

# ======== QUATERNIONS = [w,x,y,z] =======================================================================


def q_mul(q1: torch.Tensor, q2: torch.Tensor):
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].

    In:
        - q1: [B, 4]: tensor with a set of quaternions (w,x,y,z)
        - q2: [B, 4]: tensor with another set of quaternions (w,x,y,z)

    Out:
        - res: [B, 4]: tensor with the product of both quaternions
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def wrap_angle(theta: torch.Tensor):
    """
    Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
    Odd multiples of pi are wrapped to +pi (as opposed to -pi).

    In:
        - theta: [B]: tensor with a set of angles (in radians)

    Out:
        - res: [B]: tensor with the wrapped angles
    """

    pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    result[result.eq(-pi_tensor)] = math.pi

    return result


def q_angle(q: torch.Tensor):
    """
    Determine the rotation angle of given quaternion tensors of shape [*, 4].
    Return as tensor of shape [*, 1]

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B]: tensor with the corresponding rotation angles (in radians)
    """

    assert q.shape[-1] == 4

    q = q_normalize(q)

    angle = 2*torch.acos(torch.round(q[:, 0], decimals =5))

    return angle


def q_axis(q: torch.Tensor):
    """
    Determine the rotation axis of t¡he quaternion tensor of shape [*, 4].

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B, 3]: tensor with the corresponding rotation axis
    """

    assert q.shape[-1] == 4

    res = torch.zeros((q.shape[0], 3)).to(q.device)
    idx = torch.abs(torch.sin(q_angle(q = q))) > 1e-6

    res[idx] = (q[:, 1:] / torch.sin(q_angle(q = q).unsqueeze(-1) / 2))[idx]

    return res


def q_norm(q: torch.Tensor):
    """
    Compute the norm of a quaternion tensor of shape [*, 4].

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B]: tensor with the corresponding norm
    """

    assert q.shape[-1] == 4

    return torch.norm(q, dim=-1)  # ||q|| = sqrt(w²+x²+y²+z²)


def q_normalize(q: torch.Tensor):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B, 4]: tensor with the corresponding unit quaternions
    """

    assert q.shape[-1] == 4

    # Obtain the norm
    norm = q_norm(q = q)

    # check for singularities
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  
    
    # Normalize the quaternion
    return  q/norm.unsqueeze(-1)  # q_norm = q / ||q||


def q_conjugate(q: torch.Tensor):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B, 4]: tensor with the corresponding unit quaternions
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
    return q * conj.expand_as(q)


def q_is_norm(q: torch.Tensor):
    """
    Determines wether the quaternion tensor of shape [*, 4] are normalized.

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B]: tensor with boolean values indicating whether the corresponding quaternions are normalized
    """

    assert q.shape[-1] == 4

    return torch.isclose(q_norm(q = q), torch.ones(q.shape[0], device=q.device))


def q_is_pure(q: torch.Tensor):
    """
    Determines wether the quaternion tensor of shape [*, 4] is pure.

    In:
        - q: [B, 4]: tensor with a set of quaternions (w,x,y,z)

    Out:
        - res: [B]: tensor with boolean values indicating whether the corresponding quaternions are pure
    """

    assert q.shape[-1] == 4

    return torch.isclose(q[:, 0], torch.zeros(q.shape[0], device=q.device))


def q_diff(q1: torch.Tensor, q2: torch.Tensor):
    """
    Difference between two quaternions q1 and q2 of shape [*, 4].

    In:
        - q1: [B, 4]: tensor with a set of quaternions (w,x,y,z)
        - q2: [B, 4]: tensor with another set of quaternions (w,x,y,z)

    Out:
        - res: [B, 4]: tensor with the difference between both quaternions (which is another quaternion)
    """

    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    return q_mul(q1 = q_conjugate(q = q1), q2 = q2)




# ======== PURE QUATERNIONS =======================================================================


def q_inn_prod(q1: torch.Tensor, q2: torch.Tensor):
    """
    Computes the inner product between two pure quaternions q1 and q2.

    In:
        - q1: [B, 3]: tensor with a set of pure quaternions (x,y,z)
        - q2: [B, 3]: tensor with another set of pure quaternions (x,y,z)

    Out:
        - res: [B]: tensor with the corresponding inner product
    """

    assert q1.shape[-1] == 3
    assert q2.shape[-1] == 3

    return q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1] + q1[:, 2] * q2[:, 2]



# === DUAL QUATERNIONS =======================================================================


def convert_dq_to_Lab(x: torch.Tensor):
    """
    Convert a Dual Quaternion to the format of IsaacLab (translation+quaternion).

    In: 
        - x: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 7]: tensor with the corresponding translation and quaternion (x,y,z, w,x,y,z)
    """

    assert x.shape[-1] == 8

    t = dq_translation(dq = x)
    r = x[:, :4]

    return torch.cat((t, r), dim = -1)


def dq_from_tr(t: torch.Tensor, r: torch.tensor):
    """
    Create a dual quaternion from a translation tensor t of shape [*, 3] and
        a rotation quaternion tensor r of shape [*, 4].

    In: 
        - t: [B, 3]: tensor with a set of translations
        - r: [B, 4]: tensor with a set of unit quaternions (w,x,y,z)

    Out:
        - res: [B, 7]: tensor with the corresponding translation and quaternion (x,y,z, w,x,y,z)
    """

    assert r.shape[-1] == 4
    assert t.shape[-1] == 3
        
    t = torch.cat((torch.zeros(t.shape[0], 1, device= t.device), t), dim = -1)
    
    # Concatenate rotation quaternion with translation converted to quaternion
    dq = torch.cat((r, 0.5 * q_mul(t, r)), dim = -1)

    # Return normalized result
    return dq_normalize(dq = dq)


def dq_mul(dq1, dq2: torch.Tensor):
    """
    Multiply dual quaternion dq1 with dq2.
    Expects two equally-sized tensors of shape [*, 8], where * denotes any number of dimensions.
    Returns dq1*dq2 as a tensor of shape [*, 8].

    In: 
        - dq1: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)
        - dq2: [B, 8]: tensor with another set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 8]: tensor with the multiplication of both dual quaternion (w,x,y,z, w_,x_,y_,z_)
    """

    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    dq1_r, dq1_d = torch.split(dq1, [4, 4], dim=-1)
    dq2_r, dq2_d = torch.split(dq2, [4, 4], dim=-1)

    dq_prod_r = q_mul(dq1_r, dq2_r)
    dq_prod_d = q_mul(dq1_r, dq2_d) + q_mul(dq1_d, dq2_r)
    dq_prod = torch.cat([dq_prod_r, dq_prod_d], dim=-1)

    return dq_prod


def dq_translation(dq: torch.Tensor):
    """
    Returns the translation component of the input dual quaternion tensor of shape [*, 8].
    Translation is returned as tensor of shape [*, 3].

    In: 
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B]: tensor with the corresponding translation (x,y,z)
    """

    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    mult = q_mul((2.0 * dq_d), q_conjugate(dq_r))
    return mult[..., 1:]


def dq_norm(dq: torch.Tensor):
    """
    Computes the norm of a dual quaternion tensor of shape [*, 8].

    In: 
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B]: tensor with the corresponding norm (which is the norm of the primary part)
    """

    assert dq.shape[-1] == 8

    dq_r = dq[..., :4]

    return q_norm(q = dq_r)


def dq_normalize(dq: torch.Tensor):
    """
    Normalize the coefficients of a given dual quaternion tensor of shape [*, 8].

    In: 
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 8]: tensor with the normalized dual quaternions
    """

    assert dq.shape[-1] == 8

    dq_r = dq[..., :4]
    norm = torch.sqrt(torch.sum(torch.square(dq_r), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)

    # check for singularities
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=dq.device)))  
    
    return torch.div(dq, norm[:, None])  # dq_norm = dq / ||q|| = dq_r / ||dq_r|| + dq_d / ||dq_r||


def dq_conjugate(dq: torch.Tensor):
    """
    Returns the quaternion conjugate of the input dual quaternion tensor of shape [*, 8].
    The quaternion conjugate is composed of the complex conjugates of the real and the dual quaternion.

    In: 
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 8]: tensor with the corresponding conjugate
    """

    assert dq.shape[-1] == 8

    conj = torch.tensor([1, -1, -1, -1,   1, -1, -1, -1], device=dq.device)  # multiplication coefficients per element
    return dq * conj.expand_as(dq)


def dq_is_norm(dq: torch.Tensor):
    """
    Determines wether the dual quaternion tensor of shape [*, 8] is normalized.

    In: 
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B]: tensor with boolean values indicating wether the dual quaternions are normalized
    """

    assert dq.shape[-1] == 8

    return torch.isclose(dq_norm(dq = dq), torch.ones(dq.shape[0], device=dq.device))


def dq_diff(dq1, dq2: torch.Tensor):
    """"
    Computes the difference between two dual quaternions dq1 and dq2 of shape [*, 8]

    In: 
        - dq1: [B, 8]: tensor with a set of dual quaternions (w,x,y,z, w_,x_,y_,z_)
        - dq2: [B, 8]: tensor with another set of dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 8]: tensor with the difference between both tensors
    """

    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    return dq_mul(dq1 = dq_conjugate(dq = dq1), dq2 = dq2)


# === PURE DUAL QUATERNIONS =======================================================================


def dq_inn_prod(dq1: torch.Tensor, dq2: torch.Tensor):
    """
    Computes the inner product between two pure dual quaternions dq1 and dq2 of shape [*, 8].

    In: 
        - dq1: [B, 8]: tensor with a set of pure dual quaternions (w,x,y,z, w_,x_,y_,z_)
        - dq2: [B, 8]: tensor with another set of pure dual quaternions (w,x,y,z, w_,x_,y_,z_)

    Out:
        - res: [B, 8]: tensor with the inner product between both tensors
    """

    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    assert torch.any(q_is_pure(q = dq1))
    assert torch.any(q_is_pure(q = dq2))

    return -(dq_mul(dq1 = dq1, dq2 = dq2) + dq_mul(dq1 = dq2, dq2 = dq1)) / 2
