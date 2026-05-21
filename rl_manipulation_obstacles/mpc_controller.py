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





def drop_MPC_setup(obst_list, ellipsoid_r, ini = [0,0,0,0,0,0,
                                                0,0,0,0,0,0], ref = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                                                dt = 0.01,
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
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    # Horizon
    mpc.horizon = 15

    # Box constraints
    mpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10, -10, 
            -10, -10, -10, -10, -10, -10],
        x_ub=[10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10],
        u_lb=[-0.75, -0.75, -0.75, -0.01, -0.01, -0.01],
        u_ub=[0.75, 0.75, 0.75, 0.01, 0.01, 0.01],
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



def save_traj(traj, lie, saving_dir):
    dict_save = {"lie": lie,
             "traj": torch.tensor(traj) }
    
    with open(saving_dir, 'wb') as f:
        pickle.dump(dict_save, f)




# ============================================================
# UR5e DH parameters
# ============================================================

d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996


# ============================================================
# DH transform
# ============================================================

def dh(a, alpha, d, theta):

    ct = ca.cos(theta)
    st = ca.sin(theta)

    ca_ = ca.cos(alpha)
    sa_ = ca.sin(alpha)

    T = ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa_, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa_, a*st),
        ca.horzcat(0,      sa_,      ca_,    d),
        ca.horzcat(0,         0,         0,    1)
    )

    return T


# ============================================================
# Forward kinematics UR5e
# ============================================================

def ur5e_fk(q):

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]

    T1 = dh(0,      np.pi/2, d1, q1)
    T2 = dh(a2,     0,       0,  q2)
    T3 = dh(a3,     0,       0,  q3)
    T4 = dh(0,      np.pi/2, d4, q4)
    T5 = dh(0,     -np.pi/2, d5, q5)
    T6 = dh(0,      0,       d6, q6)

    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    p = T[0:3, 3]
    R = T[0:3, 0:3]

    return p, R


# ============================================================
# Orientation error
# ============================================================

def rotation_error(R, Rd):

    Re = Rd.T @ R

    e = 0.5 * ca.vertcat(
        Re[2,1] - Re[1,2],
        Re[0,2] - Re[2,0],
        Re[1,0] - Re[0,1]
    )

    return e


# ============================================================
# NMPC setup
# ============================================================

def ur5e_NMPC_setup(
    goal_position,
    goal_rotation,
    dt=0.05,
    horizon=20,
    q0=None
):

    # ========================================================
    # Create model
    # ========================================================

    model = Model(plot_backend='bokeh')

    # ========================================================
    # States (joint angles)
    # ========================================================

    q = model.set_dynamical_states([
        'q1', 'q2', 'q3',
        'q4', 'q5', 'q6'
    ])

    # ========================================================
    # Inputs (joint velocities)
    # ========================================================

    dq = model.set_inputs([
        'dq1', 'dq2', 'dq3',
        'dq4', 'dq5', 'dq6'
    ])

    # ========================================================
    # Dynamics
    # ========================================================

    dx = ca.vertcat(
        dq[0],
        dq[1],
        dq[2],
        dq[3],
        dq[4],
        dq[5]
    )

    model.set_dynamical_equations(dx)

    # ========================================================
    # Measurements
    # ========================================================

    model.set_measurements([
        'yq1', 'yq2', 'yq3',
        'yq4', 'yq5', 'yq6'
    ])

    model.set_measurement_equations(q)

    # ========================================================
    # Setup model
    # ========================================================

    model.setup(dt=dt)

    # ========================================================
    # NMPC
    # ========================================================

    nmpc = NMPC(model)

    # ========================================================
    # Forward kinematics
    # ========================================================

    p_ee, R_ee = ur5e_fk(q)

    px = p_ee[0]
    py = p_ee[1]
    pz = p_ee[2]

    # ========================================================
    # Goal
    # ========================================================

    p_ref = ca.DM(goal_position)

    R_ref = ca.DM(goal_rotation)

    # ========================================================
    # Position error
    # ========================================================

    pos_error = ca.vertcat(
        px - p_ref[0],
        py - p_ref[1],
        pz - p_ref[2]
    )

    # ========================================================
    # Orientation error
    # ========================================================

    ori_error = rotation_error(R_ee, R_ref)

    # ========================================================
    # Cost function
    # ========================================================

    stage_cost = (
        200 * ca.sumsqr(pos_error) +
        50  * ca.sumsqr(ori_error) +
        0.01 * ca.sumsqr(dq)
    )

    nmpc.set_stage_cost(stage_cost)

    # ========================================================
    # Horizon
    # ========================================================

    nmpc.horizon = horizon

    # ========================================================
    # Joint limits UR5e
    # ========================================================

    q_lb = [
        -2*np.pi,
        -2*np.pi,
        -2*np.pi,
        -2*np.pi,
        -2*np.pi,
        -2*np.pi
    ]

    q_ub = [
        2*np.pi,
        2*np.pi,
        2*np.pi,
        2*np.pi,
        2*np.pi,
        2*np.pi
    ]

    # ========================================================
    # Joint velocity limits
    # ========================================================

    dq_lb = [
        -1.5,
        -1.5,
        -1.5,
        -1.5,
        -1.5,
        -1.5
    ]

    dq_ub = [
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5
    ]

    # ========================================================
    # Constraints
    # ========================================================

    nmpc.set_box_constraints(
        x_lb=q_lb,
        x_ub=q_ub,
        u_lb=dq_lb,
        u_ub=dq_ub
    )

    # ========================================================
    # Initial condition
    # ========================================================

    if q0 is None:
        q0 = [0]*6

    u0 = [0]*6

    model.set_initial_conditions(
        x0=q0,
        z0=None
    )

    nmpc.set_initial_guess(
        x_guess=q0,
        u_guess=u0
    )

    # ========================================================
    # Setup solver
    # ========================================================

    nmpc.setup(
        options={
            'print_level': 0
        }
    )

    return model, nmpc