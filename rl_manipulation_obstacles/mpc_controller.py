import sys
sys.path.append('../../../')

from hilo_mpc import NMPC, Model
import casadi as ca
import warnings

import matplotlib.pyplot as plt
import os

import copy

import numpy as np

warnings.filterwarnings("ignore")

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle

from .mpc_utils import *



def drop_NMPC_setup(
    ini=None,
    ref=None,
    dt=0.01,
    horizon=15,
    q_min=None,
    q_max=None,
    dq_max=None,
    ddq_max=None,
    q_weights=None,
    u_weights=None,
    du_weights=None,
    T_tool=None,
    sphere_specs=None,
    n_obstacle_points=1024,
    collision_margin=0.02
):
    """
    Cartesian pose-tracking NMPC for a UR5e.

    States:
        X[:, k]: joint positions [rad].

    Inputs:
        U[:, k]: joint velocity commands [rad/s].

    Dynamics:
        q[k+1] = q[k] + dt * u[k]

    Reference pose:
        ref = [x, y, z, qw, qx, qy, qz]

    The Cartesian reference is an Opti parameter, so it can
    be updated during execution without rebuilding the solver.

    Requires:
        ur5e_fk(q, T_tool=None)
        cartesian_pose_error(q, T_ref, T_tool=None)

    Parameters
    ----------
    ini : array-like, shape (6,)
        Initial joint positions.

    ref : array-like, shape (7,)
        Initial target Cartesian pose.

    dt : float
        Control interval in seconds.

    horizon : int
        Number of control intervals.

    q_min, q_max : array-like, shape (6,)
        Joint position limits.

    dq_max : array-like, shape (6,)
        Absolute joint velocity limits.

    ddq_max : array-like, shape (6,)
        Absolute joint acceleration limits.

    q_weights : array-like, shape (6,)
        Weights for the Cartesian position and orientation errors.

    u_weights : array-like, shape (6,)
        Weights for joint velocity commands.

    du_weights : array-like, shape (6,)
        Weights for changes in joint velocity commands.

    T_tool : array-like, shape (4, 4), optional
        Fixed transformation from the robot flange to the TCP.
        If omitted, uses a 0.25 m translation along local Z,
        matching the example in the original function.
        Replace this example with the actual TCP transform.

    Returns
    -------
    controller : dict
        Contains the optimizer, decision variables, parameters,
        and compiled functions required to run the NMPC.
    """


    # --------------------------------------------------
    # 1. Parameters and defaults
    # --------------------------------------------------

    if ini is None:
        ini = [0.0] * 6

    if ref is None:
        ref = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # Copy the reference to avoid modifying the caller's array.
    ref = np.asarray(ref, dtype=float).reshape(7).copy()

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if q_min is None:
        q_min = [-2 * np.pi] * 6

    if q_max is None:
        q_max = [2 * np.pi] * 6

    if dq_max is None:
        dq_max = [2.0] * 6

    if ddq_max is None:
        ddq_max = [4.0] * 6

    if q_weights is None:
        q_weights = [1.0] * 6

    if u_weights is None:
        u_weights = [0.001] * 6

    if du_weights is None:
        du_weights = [0.001] * 6

    ini = np.asarray(ini, dtype=float).reshape(6)
    q_min = np.asarray(q_min, dtype=float).reshape(6)
    q_max = np.asarray(q_max, dtype=float).reshape(6)
    dq_max = np.asarray(dq_max, dtype=float).reshape(6)
    ddq_max = np.asarray(ddq_max, dtype=float).reshape(6)

    q_weights = np.asarray(q_weights, dtype=float).reshape(6)
    u_weights = np.asarray(u_weights, dtype=float).reshape(6)
    du_weights = np.asarray(du_weights, dtype=float).reshape(6)

    if np.any(q_min > q_max):
        raise ValueError("Every q_min element must be <= q_max.")

    if np.any(dq_max <= 0):
        raise ValueError("All dq_max values must be positive.")

    if np.any(ddq_max <= 0):
        raise ValueError("All ddq_max values must be positive.")

    if np.any(q_weights < 0) or np.any(u_weights < 0) or np.any(du_weights < 0):
        raise ValueError("Cost weights must be nonnegative.")

    Q = ca.diag(ca.DM(q_weights))
    R = ca.diag(ca.DM(u_weights))
    S = ca.diag(ca.DM(du_weights))

    # --------------------------------------------------
    # 2. Tool transformation
    # --------------------------------------------------

    if T_tool is None:
        # Example only: 25 cm along the flange's local Z axis.
        # Replace this with your actual flange-to-TCP transform.
        T_tool_np = np.eye(4)
        T_tool_np[2, 3] = 0.25
    else:
        T_tool_np = np.asarray(T_tool, dtype=float).reshape(4, 4)

    T_tool = ca.DM(T_tool_np)

    # --------------------------------------------------
    # 3. Create the optimizer and reference parameter
    # --------------------------------------------------

    opti = ca.Opti()

    # Runtime-updatable target transformation.
    T_ref = opti.parameter(4, 4)

    T_ref_initial = pose_to_transform(ref)
    opti.set_value(T_ref, T_ref_initial)

    # --------------------------------------------------
    # 4. Decision variables
    # --------------------------------------------------

    # Predicted joint positions: 6 x (N+1)
    X = opti.variable(6, horizon + 1)

    # Joint velocity commands: 6 x N
    U = opti.variable(6, horizon)

    # Parameters updated during every control iteration.
    q_measured = opti.parameter(6)
    u_previous = opti.parameter(6)

    opti.set_value(q_measured, ini)
    opti.set_value(u_previous, np.zeros(6))

    # --------------------------------------------------
    # 5. State and input constraints
    # --------------------------------------------------

    # Joint position limits
    opti.subject_to(
        opti.bounded(
            ca.repmat(ca.DM(q_min), 1, horizon + 1),
            X,
            ca.repmat(ca.DM(q_max), 1, horizon + 1),
        )
    )

    # Joint velocity limits
    opti.subject_to(
        opti.bounded(
            -ca.repmat(ca.DM(dq_max), 1, horizon),
            U,
            ca.repmat(ca.DM(dq_max), 1, horizon),
        )
    )

    # --------------------------------------------------
    # Point-cloud collision constraints
    # --------------------------------------------------

    if sphere_specs is None or len(sphere_specs) == 0:
        raise ValueError("Collision avoidance requires sphere_specs.")

    if n_obstacle_points < 1:
        raise ValueError("n_obstacle_points must be positive.")

    if not np.isfinite(collision_margin) or collision_margin < 0:
        raise ValueError("collision_margin must be finite and nonnegative.")

    # One obstacle point per column. Updated before each solve.
    obstacle_points = opti.parameter(3, n_obstacle_points)

    # Deterministic clearance; can be updated without rebuilding.
    clearance_margin = opti.parameter()
    opti.set_value(clearance_margin, collision_margin)

    # Fixed sphere centers must be checked numerically before each solve.
    fixed_sphere_indices = set()

    # Initial state is the measured joint configuration.
    opti.subject_to(X[:, 0] == q_measured)

    # --------------------------------------------------
    # 6. Cartesian tracking cost and dynamics
    # --------------------------------------------------

    J = 0

    for k in range(horizon):

        # Predicted joint configuration
        q_k = X[:, k]

        # TCP pose error, evaluated against the parameter T_ref.
        e_k, T_k = cartesian_pose_error(
            q_k,
            T_ref,
            T_tool=T_tool,
        )

        # Cartesian position and orientation tracking cost
        J_pose = ca.mtimes([e_k.T, Q, e_k])

        # Joint velocity regularization
        u_k = U[:, k]
        J_velocity = ca.mtimes([u_k.T, R, u_k])

        # Change in commanded joint velocity
        if k == 0:
            du_k = u_k - u_previous
        else:
            du_k = u_k - U[:, k - 1]

        # Smoothness cost
        J_smooth = ca.mtimes([du_k.T, S, du_k])

        # Accumulate stage cost
        J += J_pose + J_velocity + J_smooth


        # Clearance restrictions
        centers_k, radii = ur5e_collision_spheres(
            X[:, k],
            sphere_specs,
            T_tool=T_tool,
        )

        for j in range(len(sphere_specs)):

            center_j = centers_k[:, j]

            # A fixed center cannot be moved by the optimizer.
            if not ca.depends_on(center_j, ca.vec(X)):
                fixed_sphere_indices.add(j)
                continue

            delta = (
                ca.repmat(center_j, 1, n_obstacle_points)
                - obstacle_points
            )

            squared_distances = ca.sum1(delta * delta)
            required_distance = radii[j] + clearance_margin

            opti.subject_to(
                ca.vec(squared_distances)
                >= required_distance**2
            )


        # Joint-space discrete dynamics
        opti.subject_to(
            X[:, k + 1] == X[:, k] + dt * U[:, k]
        )

        # Acceleration limits:
        # |u[k] - u[k-1]| <= ddq_max * dt
        opti.subject_to(
            opti.bounded(
                -ca.DM(ddq_max) * dt,
                du_k,
                ca.DM(ddq_max) * dt,
            )
        )

    # --------------------------------------------------
    # 7. Terminal Cartesian tracking cost
    # --------------------------------------------------

    e_terminal, T_terminal = cartesian_pose_error(
        X[:, horizon],
        T_ref,
        T_tool=T_tool,
    )

    J_terminal = ca.mtimes([
        e_terminal.T,
        Q,
        e_terminal,
    ])

    J += J_terminal



        

    print(
        "Spheres requiring numerical clearance checks:",
        sorted(fixed_sphere_indices),
    )

    # --------------------------------------------------
    # 8. Objective and solver
    # --------------------------------------------------

    opti.minimize(J)

    opti.solver(
        "ipopt",
        {
            "expand": True,
            "print_time": False,
        },
        {
            "print_level": 0,
            "max_iter": 200,
            "tol": 1e-5,
        },
    )

    # --------------------------------------------------
    # 9. Compiled functions for numerical evaluation
    # --------------------------------------------------

    q_sym = ca.SX.sym("q_eval", 6)
    T_ref_sym = ca.SX.sym("T_ref_eval", 4, 4)

    e_sym, T_sym = cartesian_pose_error(
        q_sym,
        T_ref_sym,
        T_tool=T_tool,
    )

    pose_error = ca.Function(
        "pose_error",
        [q_sym, T_ref_sym],
        [e_sym],
    )

    fk_fn = ca.Function(
        "fk_fn",
        [q_sym],
        [ur5e_fk(q_sym, T_tool=T_tool)],
    )

    # All DH frame poses, packed horizontally:
    # [T_base | T_1 | ... | T_6], shape (4, 28).
    frames_sym, _ = ur5e_fk_frames(q_sym, T_tool=T_tool)

    link_poses_fn = ca.Function(
        "link_poses",
        [q_sym],
        [ca.horzcat(*frames_sym)],
    )

    # Frame origins: one column per frame, shape (3, 7).
    link_positions_fn = ca.Function(
        "link_positions",
        [q_sym],
        [ca.horzcat(*[T[:3, 3] for T in frames_sym])],
    )

    sphere_centers_fn = None
    sphere_radii = None

    if sphere_specs is not None:
        centers_sym, sphere_radii = ur5e_collision_spheres(
            q_sym,
            sphere_specs,
            T_tool=T_tool,
        )

        sphere_centers_fn = ca.Function(
            "sphere_centers",
            [q_sym],
            [centers_sym],
        )

    # --------------------------------------------------
    # 10. Return controller
    # --------------------------------------------------

    return {
        # Keep your existing entries...
        "opti": opti,
        "X": X,
        "U": U,
        "q_measured": q_measured,
        "u_previous": u_previous,
        "T_ref": T_ref,
        "T_tool": T_tool,
        "cost": J,
        "fk": fk_fn,
        "pose_error": pose_error,
        "pose_to_transform": pose_to_transform,
        "dt": dt,
        "horizon": horizon,

        # Add these:
        "link_poses": link_poses_fn,
        "link_positions": link_positions_fn,
        "sphere_centers": sphere_centers_fn,
        "sphere_radii": sphere_radii,

        "obstacle_points": obstacle_points,
        "clearance_margin": clearance_margin,
        "n_obstacle_points": n_obstacle_points,
    }




def save_traj(traj, lie, saving_dir):
    dict_save = {"lie": lie,
             "traj": torch.tensor(traj) }
    
    with open(saving_dir, 'wb') as f:
        pickle.dump(dict_save, f)

