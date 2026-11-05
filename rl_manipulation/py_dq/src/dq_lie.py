import torch
from .dq import *



def dq_adjoint(v: torch.Tensor, dq: torch.Tensor):
    """
    Adjoint for a dual quaternion dq tensor of shape [*, 8] and a twist v [*, 6].

    In:
        - v: [B, 6]: tensor with a set of twists
        - dq: [B, 8]: tensor with a set of dual quaternions (w,x,y,z,  w_,x_,y_,z_)
    
    Out: 
        - adj: [B, 6]: tensor with the adjoint
    """
    assert v.shape[-1] == 6
    assert dq.shape[-1] == 8

    device = dq.device
    
    v = torch.cat((torch.zeros(v.shape[0], 1).to(device), v[:, :3],
                   torch.zeros(v.shape[0], 1).to(device), v[:, 3:]), dim = -1)
    
    adj = dq_mul(dq1 = dq, dq2 = dq_mul(dq1 = v, dq2 = dq_conjugate(dq = dq)))

    return torch.cat((adj[:, 1:4], adj[:, 5:]), dim = -1)




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


def _q_mul(a, b):
    """Producto Hamilton. Cuaterniones en orden [w, x, y, z]."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]

    scalar = aw * bw - (av * bv).sum(dim=-1, keepdim=True)

    vector = (
        aw * bv
        + bw * av
        + torch.linalg.cross(av, bv, dim=-1)
    )

    return torch.cat((scalar, vector), dim=-1)


def _q_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)



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



def exp_bruno(dq_: torch.Tensor, kwargs=None):
    """
    Exponencial del álgebra de cuaterniones duales.

    Entrada:
        dq_: [..., 6] = [a, b].

        phi = 2*a
        rho = 2*b

    Salida:
        Q: [..., 8] = [qr, qd].
        qd = 0.5 * (0, t) ⊗ qr.
        t = J_l(phi) @ rho.

    kwargs se conserva por compatibilidad y no se utiliza.
    """
    if dq_.shape[-1] != 6:
        raise ValueError(
            "dq_ debe tener forma [...,6]"
        )

    a = dq_[..., :3]
    b = dq_[..., 3:]

    phi = 2 * a
    rho = 2 * b

    half_theta = torch.linalg.vector_norm(
        a, dim=-1, keepdim=True
    )

    qr = torch.cat(
        (
            torch.cos(half_theta),
            torch.sinc(half_theta / math.pi) * a,
        ),
        dim=-1,
    )

    _, J = _so3_exp_and_jacobian(phi)

    t = (J @ rho.unsqueeze(-1)).squeeze(-1)

    pure_t = torch.cat(
        (
            torch.zeros_like(t[..., :1]),
            t,
        ),
        dim=-1,
    )

    qd = 0.5 * _q_mul(pure_t, qr)

    return torch.cat((qr, qd), dim=-1)


def log_bruno(dq: torch.Tensor):
    """
    Logaritmo principal de un cuaternión dual.

    Entrada:
        dq: [..., 8] = [qr, qd].

    Salida:
        [..., 6] = [phi/2, rho/2].

    Asume un cuaternión dual unitario válido,
    admitiendo un factor común de escala no nulo.
    """
    if dq.shape[-1] != 8:
        raise ValueError(
            "dq debe tener forma [...,8]"
        )

    norm = torch.linalg.vector_norm(
        dq[..., :4], dim=-1, keepdim=True
    )

    if torch.any(norm == 0):
        raise ValueError(
            "El cuaternión real no puede ser nulo"
        )

    # Normalizar ambas partes con el mismo factor.
    qr = dq[..., :4] / norm
    qd = dq[..., 4:] / norm

    phi = _q_to_rotvec(qr)

    # t = 2 * vector(qd ⊗ conjugado(qr)).
    t = 2 * _q_mul(qd, _q_conj(qr))[..., 1:]

    _, J = _so3_exp_and_jacobian(phi)

    rho = torch.linalg.solve(
        J,
        t.unsqueeze(-1),
    ).squeeze(-1)

    return torch.cat((phi, rho), dim=-1)


def exp_stereo_q(q_: torch.Tensor):
    """
    Stereographic exponential map for a pure quaternion of shape [*, 3].

    In: 
        - q_: [B, 3]: tensor with a set of pure quaternions without the real part (x,y,z)
    
    Out: 
        - res: [B, 4]: tensor with the corresponding unit quaternions (w,x,y,z)
    """

    assert q_.shape[-1] == 3

    # Denominator
    div = (1 + torch.norm(q_, dim = -1)).unsqueeze(-1)

    # Real part and imaginary part
    re = (1 - torch.pow(torch.norm(q_, dim = -1), 2)).unsqueeze(-1) / div
    im = 2*q_ / div

    # Return the concatenation of both parts
    return torch.cat((re, im), dim = -1)


def log_stereo_q(q: torch.Tensor):
    """
    Stereographic logarithmic map for a unit quaternion of shape [*, 4].

    In: 
        - q: [B, 4]: tensor with a set of unit quaternions (w,x,y,z)
    
    Out: 
        - res: [B, 3]: tensor with the corresponding pure quaternions (x,y,z)
    """
    assert q.shape[-1] == 4

    return q[:, 1:] / (1+q[:, 0]).unsqueeze(-1)


def exp_stereo(dq_:torch.Tensor, kwargs = None):
    """
    Stereographic exponential map for a dual quaternion of shape [*, 8] -> it is the same as the map for a quaternion+translation.

    In: 
        - dq_: [B, 6]: tensor with a set of twists from the dual quaternion group
    
    Out: 
        - res: [B, 8]: tensor with the corresponding unit dual quaternions quaternions (w,x,y,z, w_,x_,y_,z_)
    """

    assert dq_.shape[-1] == 6
    
    # Obtain the exponential of the primary part with the stereographic map and ...
    q_p = exp_stereo_q(q_ = dq_[:, :3])  

    # ... the translation from the dual part
    res = dq_from_tr(t = dq_[:, 3:], r = q_p)

    # Fix dual cover
    neg_idx = res[:, 0] < 0.0
    res[neg_idx] *= -1

    return res


def log_stereo(dq: torch.Tensor):
    """
    Stereographic logarithmic map for a dual quaternion of shape [*, 8] -> is the same as the map for a quaternion+translation.

    In: 
        - dq: [B, 8]: tensor with a set of unit dual quaternions (w,x,y,z, w_,x_,y_,z_)
    
    Out: 
        - res: [B, 6]: tensor with the corresponding twists
    """

    assert dq.shape[-1] == 8
    assert torch.any(dq_is_norm(dq))

    # Fix dual cover
    neg_idx = dq[:, 0] < 0.0
    dq[neg_idx] *= -1

    # Concatenate the logarithm of the primary part with the translation
    return torch.cat((log_stereo_q(q = dq[:, :4]), 
                      dq_translation(dq = dq)), dim = -1)

