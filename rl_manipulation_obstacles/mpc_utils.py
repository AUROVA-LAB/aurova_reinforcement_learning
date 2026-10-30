import casadi as ca
import numpy as np



def pose_to_transform(ref):
    """
    Convert [x, y, z, qw, qx, qy, qz] into a 4x4
    homogeneous transformation matrix.
    """
    ref = np.asarray(ref, dtype=float).reshape(7)

    px, py, pz, qw, qx, qy, qz = ref
    px *= -1

    # Normalize quaternion
    # q_norm = np.linalg.norm([qw, qx, qy, qz])
    # if q_norm < 1e-12:
    #     raise ValueError("Reference quaternion must not be zero.")

    # qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]

    return T


def ur5e_fk(q, T_tool=None):
    """
    UR5e forward kinematics using standard DH parameters.

    Parameters
    ----------
    q : casadi.SX, MX, or DM
        Six joint positions in radians.
    T_tool : optional
        4x4 transform from the flange to the TCP/tool frame.

    Returns
    -------
    T : casadi 4x4 matrix
        Transformation from the base frame to the TCP.
    """

    # Standard DH parameters for UR5e (meters, radians).
    a = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
    d = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
    alpha = [
        np.pi / 2,
        0.0,
        0.0,
        np.pi / 2,
        -np.pi / 2,
        0.0,
    ]

    T = ca.SX.eye(4) if isinstance(q, ca.SX) else ca.MX.eye(4)

    for i in range(6):
        theta = q[i]

        ct = ca.cos(theta)
        st = ca.sin(theta)
        ca_i = np.cos(alpha[i])
        sa_i = np.sin(alpha[i])

        A_i = ca.vertcat(
            ca.horzcat(
                ct, -st * ca_i, st * sa_i, a[i] * ct
            ),
            ca.horzcat(
                st, ct * ca_i, -ct * sa_i, a[i] * st
            ),
            ca.horzcat(
                0, sa_i, ca_i, d[i]
            ),
            ca.horzcat(
                0, 0, 0, 1
            ),
        )

        T = T @ A_i

    if T_tool is not None:
        T = T @ T_tool

    return T



def hat(v):
    """
    Convert a 3D vector into a skew-symmetric matrix.
    """
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def vee(A):
    """
    Extract a 3D vector from a skew-symmetric matrix.
    """
    return ca.vertcat(
        A[2, 1],
        A[0, 2],
        A[1, 0],
    )


def rotation_log(R):
    """
    SO(3) logarithm map.

    Parameters
    ----------
    R : CasADi SX, MX, or DM, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    phi : CasADi vector, shape (3, 1)
        Rotation vector in radians.

    Notes
    -----
    Intended for optimization near the target.
    Rounding is omitted to preserve differentiability.
    """

    # Cosine of the rotation angle.
    cos_theta = (ca.trace(R) - 1.0) / 2.0
    cos_theta = ca.fmin(1.0, ca.fmax(-1.0, cos_theta))

    # Skew-symmetric component.
    K = 0.5 * (R - R.T)

    # vee(K) = sin(theta) * rotation_axis.
    phi_sin = vee(K)

    # Norm of the sine-axis vector.
    sin_theta = ca.sqrt(ca.sumsqr(phi_sin) + 1e-24)

    # Rotation angle.
    theta = ca.atan2(sin_theta, cos_theta)

    # Small-angle handling.
    is_small = theta < 1e-6

    # theta / sin(theta), with a safe denominator.
    sin_theta_safe = ca.fmax(sin_theta, 1e-12)
    scale_regular = theta / sin_theta_safe

    # Taylor expansion near zero.
    scale_small = 1.0 + theta**2 / 6.0

    scale = ca.if_else(
        is_small,
        scale_small,
        scale_regular,
    )

    # Rotation vector.
    phi = scale * phi_sin
    
    return phi

def cartesian_pose_error(q, T_ref, T_tool=None):
    """
    Compute Cartesian position and orientation errors for a UR5e.

    Parameters
    ----------
    q : CasADi vector, shape (6, 1)
        Joint positions in radians.

    T_ref : CasADi matrix or numeric array, shape (4, 4)
        Desired TCP pose in the same base frame as ur5e_fk().

    T_tool : optional
        Flange-to-TCP transformation passed to ur5e_fk().

    Returns
    -------
    e : CasADi vector, shape (6, 1)
        [position error (m), orientation error (rad)].

        Both errors are expressed in the base frame.

    T : CasADi matrix, shape (4, 4)
        Current TCP transformation.
    """

    # Convert numeric reference poses into CasADi matrices.
    # Existing CasADi expressions are left unchanged.
    if not isinstance(T_ref, (ca.SX, ca.MX, ca.DM)):
        T_ref = ca.DM(T_ref)

    # Forward kinematics.
    T = ur5e_fk(q, T_tool)

    # Current TCP pose.
    p = T[0:3, 3]
    R = T[0:3, 0:3]

    # Reference TCP pose.
    p_ref = T_ref[0:3, 3]
    R_ref = T_ref[0:3, 0:3]

    # Position error, expressed in the base frame.
    e_p = p - p_ref

    # Orientation error, expressed in the base frame.
    R_err = R_ref @ R.T

    e_R = rotation_log(R_err)

    # Combined Cartesian pose error.
    e = ca.vertcat(e_p, e_R)

    return e, T