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





def drop_MPC_setup(obst_list, ellipsoid_r, ini = [0,0,0,0,0,0,
                                                0,0,0,0,0,0], ref = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                                                dt = 0.01,
                                                limits = [[-0.75, -0.75, -0.75, -0.01, -0.01, -0.01],
                                                          [0.75, 0.75, 0.75, 0.01, 0.01, 0.01]],
                                                lie = False,):
        
    # ======================================================
    # Create model
    # ======================================================
    model = Model(plot_backend='bokeh')

    # States
    x = model.set_dynamical_states(['X', 'Y',    'Z', 'X_', 'Y_', 'Z_', 
                                    'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'])
    X = x[0]
    Y = x[1]
    Z = x[2]
    X_ = x[3]
    Y_ = x[4]
    Z_ = x[5]

    Vx = x[6]
    Vy = x[7]
    Vz = x[8]
    Wx = x[9]
    Wy = x[10]
    Wz = x[11]

    # Measurements
    model.set_measurements(['yX', 'yY', 'yZ', 'yX_', 'yY_', 'yZ_',
                            'yVx', 'yVy', 'yVz', 'yWx', 'yWy', 'yWz'])
    model.set_measurement_equations(x)

    # Inputs
    u = model.set_inputs(['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'])
    ax = u[0]
    ay = u[1]
    az = u[2]
    ax_ = u[3]
    ay_ = u[4]
    az_ = u[5]

    # ======================================================
    # Dynamics
    # ======================================================
    dx = ca.vertcat(
        Vx,
        Vy,
        Vz,
        Wx,
        Wy,
        Wz,
        ax,
        ay,
        az,
        ax_,
        ay_,
        az_,
    )
    model.set_dynamical_equations(dx)


    # ======================================================
    # Obstacle avoidance via algebraic constraint
    # ======================================================




    rhs = []


    n_obst = obst_list.shape[-1]

    idx_obst = [[0, 1, 2], [2, 0, 1]]

    ellipsoid_r_torch = []

    if n_obst > 0:
        z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(obst_list))])
        
        # Obstacle parameters
        for idx, o in enumerate(obst_list):

            # Algebraic state (constraint slack, optional)

            # Constraint equation: c_obs = ((X-Xo)/a)^2 + ((Y-Yo)/b)^2 + ((Z - Zo)/c)^2- 1 ->
            '''
            sería algo así como:
                rhs = (z = ((X-Xo)/a)^2 + ((Y-Yo)/b)^2 + ((Z - Zo)/c)^2- 1)

            pero lo tienes que poner de manera que rhs = 0 para que entre en "set_algebraic_equations" 
            '''

            change_idx = idx >= int(n_obst / 2)

            rhs.append((X - o[0].item())**2 / ((-0.25/2 + 0.01 + ellipsoid_r[idx_obst[change_idx][0]]))**2 + \
                    (Y - o[1].item())**2 / ((0.01 + ellipsoid_r[idx_obst[change_idx][1]] + 1.0*change_idx))**2 + \
                    (Z - o[2].item())**2 / ((0.01 + ellipsoid_r[idx_obst[change_idx][2]] + 1.0*(not change_idx)))**2 - 1 - z[idx])
            
            ellipsoid_r_torch.append([-0.25/2 + ellipsoid_r[idx_obst[change_idx][0]], 
                                    1.0*change_idx + ellipsoid_r[idx_obst[change_idx][1]], 
                                    1.0*(not change_idx) + ellipsoid_r[idx_obst[change_idx][2]]])
        
        ellipsoid_r_torch = torch.tensor(ellipsoid_r_torch)
        model.set_algebraic_equations(ca.vertcat(*rhs))




    # ======================================================
    # Setup model
    # ======================================================
    model.setup(dt=dt)

    # ======================================================
    # NMPC
    # ======================================================
    mpc = NMPC(model)

    # Target
    X_ref =  ref[0].item()
    Y_ref =  ref[1].item()
    Z_ref =  ref[2].item()
    X__ref = ref[3].item()
    Y__ref = ref[4].item()
    Z__ref = ref[5].item()

    Vx_ref = 0.0
    Vy_ref = 0.0
    Vz_ref = 0.0
    Wx_ref = 0.0
    Wy_ref = 0.0
    Wz_ref = 0.0

    weights = [[50, 70, 60, 50, 50, 50,     2, 2, 2, 2, 2, 2], 
               [3, 3, 3, 3, 3, 3,     3, 3, 3, 3, 3, 3]]

    mpc.quad_stage_cost.add_states(
        names=['X', 'Y', 'Z', 'X_', 'Y_', 'Z_', 
                'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'],

        ref=[X_ref, Y_ref, Z_ref, X__ref, Y__ref, Z__ref,
                Vx_ref, Vy_ref, Vz_ref, Wx_ref, Wy_ref, Wz_ref],
        weights=weights[lie]
    )

    mpc.quad_stage_cost.add_inputs(
        names=['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'],
        weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Horizon
    mpc.horizon = 15

    # Box constraints
    mpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10, -10, 
            -10, -10, -10, -10, -10, -10],
        x_ub=[10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10],
        u_lb=limits[0],# [-0.75, -0.75, -0.75, -0.01, -0.01, -0.01],
        u_ub=limits[1],# [0.75, 0.75, 0.75, 0.01, 0.01, 0.01],
        z_lb=None, # [0.0]*n_obst,      # <-- enforces obstacle avoidance
        z_ub=None, # [ca.inf]*n_obst
    )

    # Initial conditions
    x0 = ini
    # z0 = [1.0]*len(obst_list)   # start feasible
    u0 = [0]*6
    
    model.set_initial_conditions(x0=x0, z0 = None)
    mpc.set_initial_guess(x_guess=x0, u_guess=u0)

    mpc.setup(options={'print_level': 0})


    return model, mpc, ellipsoid_r_torch





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
):
    """
    Cartesian pose-tracking NMPC for a UR5e.

    States:
        q: Joint positions [rad].

    Inputs:
        u: Joint velocity commands [rad/s].

    Dynamics:
        q[k+1] = q[k] + dt*u[k]

    Returns:
        controller: dictionary containing the optimizer and
                    the functions required to run the MPC.

    Requires the previously defined:
        ur5e_fk(q, T_tool=None)
        cartesian_pose_error(q, T_ref, T_tool=None)
    """

    # --------------------------------------------------
    # 1. Parameters
    # --------------------------------------------------

    if ini is None:
        ini = [0.0] * 6

    if ref is None:
        ref = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]

    if q_min is None:
        q_min = [-2 * np.pi] * 6

    if q_max is None:
        q_max = [2 * np.pi] * 6

    if dq_max is None:
        dq_max = [1.0] * 6

    if ddq_max is None:
        ddq_max = [2.0] * 6

    if q_weights is None:
        q_weights = [100.0, 100.0, 100.0,
                     10.0, 10.0, 10.0]

    if u_weights is None:
        u_weights = [0.01] * 6

    if du_weights is None:
        du_weights = [0.1] * 6

    ini = np.asarray(ini, dtype=float).reshape(6)
    ref = np.asarray(ref, dtype=float).reshape(6)

    q_min = np.asarray(q_min, dtype=float).reshape(6)
    q_max = np.asarray(q_max, dtype=float).reshape(6)
    dq_max = np.asarray(dq_max, dtype=float).reshape(6)
    ddq_max = np.asarray(ddq_max, dtype=float).reshape(6)

    Q = ca.diag(ca.DM(q_weights))
    R = ca.diag(ca.DM(u_weights))
    S = ca.diag(ca.DM(du_weights))

    # --------------------------------------------------
    # 2. Target pose
    # --------------------------------------------------

    # ref = [x, y, z, roll, pitch, yaw]
    #
    # Target orientation uses Rz(yaw)*Ry(pitch)*Rx(roll).
    # Angles are in radians.

    px, py, pz, roll, pitch, yaw = ref

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    R_ref = Rz @ Ry @ Rx

    T_ref_np = np.eye(4)
    T_ref_np[:3, :3] = R_ref
    T_ref_np[:3, 3] = [px, py, pz]

    T_ref = ca.DM(T_ref_np)

    # Symbolic joint configuration
    q_sym = ca.SX.sym("q_eval", 6)

    # Symbolic Cartesian error
    e_sym, T_sym = cartesian_pose_error(q_sym, T_ref)

    # Compiled CasADi functions for numerical evaluation
    # TODO: estas funciones llaman a cartesian_pose_error pero no incluyen T_tool
    pose_error_fn = ca.Function(
        "pose_error_fn",
        [q_sym],
        [e_sym],
    )

    fk_fn = ca.Function(
        "fk_fn",
        [q_sym],
        [T_sym],
    )

    # --------------------------------------------------
    # 3. Optimization problem
    # --------------------------------------------------

    opti = ca.Opti()

    # Predicted joint positions: 6 x (N+1)
    X = opti.variable(6, horizon+1)

    # Joint velocity commands: 6 x N
    U = opti.variable(6, horizon)

    # Parameters updated at every control iteration
    q_measured = opti.parameter(6)
    u_previous = opti.parameter(6)

    opti.set_value(q_measured, ini)
    opti.set_value(u_previous, np.zeros(6))

    # --------------------------------------------------
    # 4. State and input constraints
    # --------------------------------------------------

    # Joint position limits
    opti.subject_to(
        opti.bounded(
            ca.repmat(ca.DM(q_min), 1, horizon+1),
            X,
            ca.repmat(ca.DM(q_max), 1, horizon+1),
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

    # Initial measured joint positions
    opti.subject_to(X[:, 0] == q_measured)

    # --------------------------------------------------
    # 5. Symbolic Cartesian cost
    # --------------------------------------------------

    J = 0

    for k in range(horizon - 1):

        # Predicted joint configuration
        q_k = X[:, k]

        # Forward kinematics and Cartesian pose error
        e_k, T_k = cartesian_pose_error(q_k, T_ref)

        # This is the symbolic Cartesian cost.
        J_pose = ca.mtimes([e_k.T, Q, e_k])

        # Joint velocity regularization
        u_k = U[:, k]
        J_velocity = ca.mtimes([u_k.T, R, u_k])

        # Velocity change regularization
        if k == 0:
            du_k = u_k - u_previous
        else:
            du_k = u_k - U[:, k - 1]

        J_smooth = ca.mtimes([du_k.T, S, du_k])

        # Total stage cost
        J += J_pose + J_velocity + J_smooth

        # --------------------------------------------------
        # 6. Joint-space dynamics
        # --------------------------------------------------

        opti.subject_to(
            X[:, k + 1] == X[:, k] + dt * U[:, k]
        )

        # --------------------------------------------------
        # 7. Joint acceleration constraints
        # --------------------------------------------------

        opti.subject_to(
            opti.bounded(
                -ddq_max * dt,
                du_k,
                ddq_max * dt,
            )
        )

    # --------------------------------------------------
    # 8. Terminal pose cost
    # --------------------------------------------------

    e_terminal, T_terminal = cartesian_pose_error(
        X[:, horizon - 1],
        T_ref,
    )

    J_terminal = ca.mtimes([
        e_terminal.T,
        Q,
        e_terminal,
    ])

    J += J_terminal

    # --------------------------------------------------
    # 9. Set objective and solver
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
    # 10. Return controller
    # --------------------------------------------------

    return {
        "opti": opti,
        "X": X,
        "U": U,
        "q_measured": q_measured,
        "u_previous": u_previous,
        "T_ref": T_ref,
        "cost": J,
        "fk": fk_fn,
        "pose_error": pose_error_fn,
        "dt": dt,
        "horizon": horizon,
    }




def save_traj(traj, lie, saving_dir):
    dict_save = {"lie": lie,
             "traj": torch.tensor(traj) }
    
    with open(saving_dir, 'wb') as f:
        pickle.dump(dict_save, f)


