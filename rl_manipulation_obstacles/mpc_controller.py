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





def drop_NMPC_setup(obst_list, 
                    ellipsoid_r, 
                    ini = [0,0,0,0,0,0,
                           0,0,0,0,0,0], 
                    ref = [1.0, 1.0, 1.0, 
                           0.0, 0.0, 0.0], 
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

    n_obst = obst_list.shape[0]

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

            rhs.append((X - o[0].item())**2   / ((ellipsoid_r[0]))**2 + \
                       (Y - o[1].item())**2   / ((ellipsoid_r[1]))**2 + \
                       (Z - o[2].item())**2   / ((ellipsoid_r[2]))**2 - 1 - z[idx])
            
            ellipsoid_r_torch.append([ellipsoid_r[0], 
                                      ellipsoid_r[1], 
                                      ellipsoid_r[2]])
        
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
    nmpc.horizon = 15

    # print(obst_list)
    # print(n_obst)
    # print("\n\n\n")

    # Box constraints
    nmpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10, -10, 
            -10, -10, -10, -10, -10, -10],
        x_ub=[10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10],
        u_lb=[-0.75, -0.75, -0.75, -0.01, -0.01, -0.01],
        u_ub=[0.75, 0.75, 0.75, 0.01, 0.01, 0.01],
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