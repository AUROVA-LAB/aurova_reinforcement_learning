import sys
sys.path.append('../../../')

from hilo_mpc import NMPC, Model
import casadi as ca
import warnings


warnings.filterwarnings("ignore")

import torch
import pickle
import math





def drop_NMPC_setup(obst_list, 
                    ellipsoid_r, 
                    ini = [0,0,0,0,0,0,
                           0,0,0,0,0,0], 
                    ref = [1.0, 1.0, 1.0, 
                           0.0, 0.0, 0.0], 
                    dt = 0.01,
                    lie = False,
                    obst_rot = [0.0, 0.0, 0.0]):
        
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
    ellipsoid_r_torch = []
    n_obst = obst_list.shape[0]

    theta_roll, theta_pitch, theta_yaw = obst_rot[0], obst_rot[1], obst_rot[2]

    if n_obst > 0:
        z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(obst_list))])

        for idx, o in enumerate(obst_list):

            # Center of obstacle
            o_vec = ca.DM([o[0].item(), o[1].item(), o[2].item()])

            # Semi-axes
            a = ellipsoid_r[0]
            b = ellipsoid_r[1]
            c = ellipsoid_r[2]

            # -----------------------------
            # Rotation (replace these with your stored angles)
            # -----------------------------
            roll  = theta_roll[idx] if isinstance(theta_roll, (list, tuple)) else 0.0
            pitch = theta_pitch[idx] if isinstance(theta_pitch, (list, tuple)) else 0.0
            yaw   = theta_yaw[idx] if isinstance(theta_yaw, (list, tuple)) else 0.0

            Rz = ca.vertcat(
                ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
                ca.horzcat(ca.sin(yaw),  ca.cos(yaw), 0),
                ca.horzcat(0, 0, 1)
            )

            Ry = ca.vertcat(
                ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
                ca.horzcat(0, 1, 0),
                ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
            )

            Rx = ca.vertcat(
                ca.horzcat(1, 0, 0),
                ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
                ca.horzcat(0, ca.sin(roll),  ca.cos(roll))
            )

            R = Rz @ Ry @ Rx

            # -----------------------------
            # Rotated ellipsoid constraint
            # -----------------------------
            p = ca.vertcat(X_, Y_, Z_)
            d = p - o_vec

            d_rot = R @ d

            rhs.append(
                (d_rot[0] / a)**2 +
                (d_rot[1] / b)**2 +
                (d_rot[2] / c)**2
                - 1 - z[idx]
            )

            ellipsoid_r_torch.append([a, b, c])

        ellipsoid_r_torch = torch.tensor(ellipsoid_r_torch)
        model.set_algebraic_equations(ca.vertcat(*rhs))




    # ======================================================
    # Setup model
    # ======================================================
    model.setup(dt=dt)

    # ======================================================
    # NMPC
    # ======================================================
    nmpc = NMPC(model)

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
    
    nmpc.quad_stage_cost.add_states(
        names=['X', 'Y', 'Z', 'X_', 'Y_', 'Z_', 
                'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'],

        ref=[X_ref, Y_ref, Z_ref, X__ref, Y__ref, Z__ref,
                Vx_ref, Vy_ref, Vz_ref, Wx_ref, Wy_ref, Wz_ref],
        weights=weights[lie]
    )

    nmpc.quad_stage_cost.add_inputs(
        names=['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'],
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    # Horizon
    nmpc.horizon = 45

    # print(obst_list)
    # print(n_obst)
    # print("\n\n\n")

    # Box constraints
    nmpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10,  0,      -10, -10, -10, -10, -10, -10],
        x_ub=[ 10,  10,  10,   0,  10, 10,       10,  10,  10,  10,  10,  10],

        u_lb=[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
        u_ub=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        z_lb=[0.0]*n_obst,      # <-- enforces obstacle avoidance
        z_ub=[ca.inf]*n_obst
    )

    # Initial conditions
    x0 = ini
    z0 = [1.0]*n_obst  # start feasible
    u0 = [0]*6

    model.set_initial_conditions(x0=x0, z0 = z0)
    nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

    nmpc.setup(options={'print_level': 0})


    return model, nmpc, ellipsoid_r_torch

def save_traj(traj, lie, saving_dir):
    dict_save = {"lie": lie,
             "traj": torch.tensor(traj) }
    
    with open(saving_dir, 'wb') as f:
        pickle.dump(dict_save, f)