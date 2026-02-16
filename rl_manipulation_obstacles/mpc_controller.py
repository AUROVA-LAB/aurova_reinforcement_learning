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





def drop_NMPC_setup(obst_list, ellipsoid_r, ini = [0,0,0,0,0,0,
                                                0,0,0,0,0,0], ref = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                                                lie = False):
        
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


    z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(obst_list))])


    rhs = []


    n_obst = len(obst_list)

    idx_obst = [[0, 1, 2], [2, 0, 1]]

    ellipsoid_r_torch = []

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
    dt = 0.01
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
               [50, 50, 50, 50, 70, 60,     2, 2, 2, 2, 2, 2]]

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
    nmpc.horizon = 15

    # Box constraints
    nmpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10, -10, 
            -10, -10, -10, -10, -10, -10],
        x_ub=[10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10],
        u_lb=[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        u_ub=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        z_lb=[0.0]*len(obst_list),      # <-- enforces obstacle avoidance
        z_ub=[ca.inf]*len(obst_list)
    )

    # Initial conditions
    x0 = ini.cpu().numpy().tolist()
    z0 = [1.0]*len(obst_list)   # start feasible
    u0 = [0]*6

    model.set_initial_conditions(x0=x0, z0=z0)
    nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

    nmpc.setup(options={'print_level': 0})


    return model, nmpc, ellipsoid_r_torch

def save_traj(traj, lie):
    dict_save = {"lie": lie,
             "traj": torch.tensor(traj) }
    with open('./rl_manipulation_obstacles/traj.pkl', 'wb') as f:
        pickle.dump(dict_save, f)