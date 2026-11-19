import casadi as ca
import numpy as np

def pose_to_transform(ref):
    """
    Convert [x, y, z, qw, qx, qy, qz] into a homogeneous transform.

    No change of reference frame is performed.
    """
    ref = np.asarray(ref, dtype=float).reshape(7)

    if not np.all(np.isfinite(ref)):
        raise ValueError("Reference pose must contain finite values.")

    px, py, pz = ref[:3]
    quat = ref[3:]

    q_norm = np.linalg.norm(quat)
    if q_norm < 1e-12:
        raise ValueError("Reference quaternion must not be zero.")

    qw, qx, qy, qz = quat / q_norm

    R = np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx*qy - qw*qz),
            2 * (qx*qz + qw*qy),
        ],
        [
            2 * (qx*qy + qw*qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy*qz - qw*qx),
        ],
        [
            2 * (qx*qz - qw*qy),
            2 * (qy*qz + qw*qx),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]

    return T

def ur5e_fk_frames(q, T_tool=None):
    """
    Return base-to-DH-frame transformations.

    frames[0]: identity, robot base
    frames[1]: after joint 1
    ...
    frames[6]: after joint 6

    T_tcp: final transform including T_tool, if provided.

    IMPORTANT:
        These are DH frames. Their origins/orientations may differ
        from the named link frames in Isaac Lab / the URDF.
    """
    q = ca.vertcat(q[0] + np.pi, q[1:])

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

    # Start with a numeric identity; CasADi promotes it as needed.
    T = ca.DM.eye(4)
    frames = [T]

    for i in range(6):
        ct = ca.cos(q[i])
        st = ca.sin(q[i])
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
        frames.append(T)

    T_tcp = T
    if T_tool is not None:
        T_tcp = T_tcp @ ca.DM(T_tool)

    return frames, T_tcp

def ur5e_fk(q, T_tool=None):
    """Existing interface, retained for your pose-tracking code."""
    _, T_tcp = ur5e_fk_frames(q, T_tool=T_tool)
    return T_tcp

def ur5e_collision_spheres(q, sphere_specs, T_tool=None):
    """
    Compute collision-sphere centers in the robot base frame.

    sphere_specs:
        Sequence of dictionaries with:
            frame: integer 0..6, or "tcp"
            center: local [x, y, z] in metres
            radius: positive radius in metres

    Returns
    -------
    centers : CasADi matrix, shape (3, number_of_spheres)
    radii   : CasADi DM, shape (number_of_spheres, 1)
    """
    if len(sphere_specs) == 0:
        raise ValueError("Provide at least one collision sphere.")

    frames, T_tcp = ur5e_fk_frames(q, T_tool=T_tool)

    centers = []
    radii = []

    for spec in sphere_specs:
        frame = spec["frame"]

        if frame == "tcp":
            T = T_tcp
        elif isinstance(frame, (int, np.integer)) and 0 <= frame <= 6:
            T = frames[int(frame)]
        else:
            raise ValueError("Sphere frame must be 0..6 or 'tcp'.")

        center_local = np.asarray(
            spec["center"], dtype=float
        ).reshape(3)

        radius = float(spec["radius"])

        if not np.all(np.isfinite(center_local)):
            raise ValueError("Sphere center must be finite.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Sphere radius must be finite and positive.")

        center_base = (
            T[:3, :3] @ ca.DM(center_local)
            + T[:3, 3]
        )

        centers.append(center_base)
        radii.append(radius)

    return ca.horzcat(*centers), ca.DM(radii)

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