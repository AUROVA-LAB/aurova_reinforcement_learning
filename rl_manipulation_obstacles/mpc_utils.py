import casadi as ca
import numpy as np


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


def rotation_log(R):
    """
    SO(3) logarithm, returning a 3-vector.

    R is a relative rotation matrix.

    The implementation is intended for optimization near the
    target, where the orientation error is generally small.
    """

    R_skew = 0.5 * (R - R.T)

    vee = ca.vertcat(
        R_skew[2, 1],
        R_skew[0, 2],
        R_skew[1, 0],
    )

    # vee = sin(theta) * axis
    sin_theta = ca.sqrt(
        ca.sumsqr(vee) + 1e-12
    )

    cos_theta = (
        ca.trace(R) - 1.0
    ) / 2.0

    # atan2 is preferable to acos for numerical stability.
    theta = ca.atan2(sin_theta, cos_theta)

    # Log(R) = theta / sin(theta) * vee.
    # Near zero, use the first-order approximation.
    scale = ca.if_else(
        sin_theta < 1e-5,
        1.0,
        theta / sin_theta,
    )

    return scale * vee


def cartesian_pose_error(q, T_ref, T_tool=None):
    """
    Compute the Cartesian pose error for a UR5e.

    Parameters
    ----------
    q : CasADi vector, shape (6,)
        Joint positions in radians.
    T_ref : CasADi matrix or numeric 4x4 matrix
        Desired TCP pose in the same base frame as ur5e_fk().
    T_tool : optional
        Flange-to-TCP transform.

    Returns
    -------
    e : CasADi vector, shape (6,)
        [position error (m), orientation error (rad)]
    T : CasADi 4x4 matrix
        Current TCP transformation.
    """

    T = ur5e_fk(q, T_tool)

    p = T[0:3, 3]
    R = T[0:3, 0:3]

    p_ref = T_ref[0:3, 3]
    R_ref = T_ref[0:3, 0:3]

    # Position error in the base frame.
    e_p = p - p_ref

    # Relative orientation error.
    R_err = R_ref.T @ R
    e_R = rotation_log(R_err)

    e = ca.vertcat(e_p, e_R)

    return e, T