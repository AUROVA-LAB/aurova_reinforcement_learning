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

def get_frame(x, tgt, obst, ax):
    

    X, Y = float(x[0]), float(x[1])
    X_tgt, Y_tgt = tgt[0], tgt[1]

    margin = 1.0

    xs = [X, X_tgt, obst[0], 0]
    ys = [Y, Y_tgt, obst[1], 0]

    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)




    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        ax.cla()

    # Plot robot position
    ax.scatter(X, Y, s=80, c='k', zorder=3)

    # Plot goal position
    theta = np.linspace(0, 2*np.pi, 200)
    xe = Xo + 0.3 * np.cos(theta)
    ye = Yo + 0.15 * np.sin(theta)
    ax.plot(xe, ye, 'b--', linewidth=2)

    ax.scatter(X_tgt, Y_tgt, s = 80, c='g', zorder = 3)
    ax.scatter(obst[0], obst[1], s = 80, c='b', zorder = 3)

    # Optional: draw origin
    ax.scatter(0, 0, c='r', s=80, zorder=3)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Robot Position")

    return fig, ax


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

# Shelf poses
p1 = [-0.75, -0.6, 0.25, 1,0,0,0]
p2 = [-0.75, -0.35, 0.0, 1,0,0,0]

shelf_poses = [p1, p2]

for i in range(2):
    p_ = copy.deepcopy(shelf_poses[i])
    
    for j in range(4):

        if j != 0:
            p_[1] += 0.5
            shelf_poses.append(copy.deepcopy(p_))
        
        p__ = copy.deepcopy(p_)

        for k in range(2):
            p__[2] += 0.5
            shelf_poses.append(copy.deepcopy(p__))


ellipsoid_r = [0.15, 0.05, 0.25]

z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(shelf_poses))])


rhs = []
# Obstacle parameters
for idx, o in enumerate(shelf_poses):

    # Algebraic state (constraint slack, optional)

    # Constraint equation: c_obs = (X-Xo)^2 + (Y-Yo)^2 - r^2 ->
    '''
    sería algo así como:
        rhs = (z = (X-Xo)^2 + (Y-Yo)^2 - r^2)

    pero lo tienes que poner de manera que rhs = 0 para que entre en "set_algebraic_equations" 
    '''
    
    rhs.append((X - o[0])**2 / ellipsoid_r[0]**2 + (Y - o[1])**2 / ellipsoid_r[1]**2 + (Z - o[2])**2 / ellipsoid_r[2]**2 - 1 - z[idx])

model.set_algebraic_equations(ca.vertcat(*rhs))






# ======================================================
# Setup model
# ======================================================
dt = 0.1
model.setup(dt=dt)

# ======================================================
# NMPC
# ======================================================
nmpc = NMPC(model)

# Target
X_ref = 1.0
Y_ref = 1.0
Z_ref = 1.0
X__ref = 1.0
Y__ref = 1.0
Z__ref = 1.0

Vx_ref = 0.0
Vy_ref = 0.0
Vz_ref = 0.0
Wx_ref = 0.0
Wy_ref = 0.0
Wz_ref = 0.0

nmpc.quad_stage_cost.add_states(
    names=['X', 'Y', 'Z', 'X_', 'Y_', 'Z_', 
            'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'],

    ref=[X_ref, Y_ref, Z_ref, X__ref, Y__ref, Z__ref,
            Vx_ref, Vy_ref, Vz_ref, Wx_ref, Wy_ref, Wz_ref],
    weights=[50, 50, 50, 50, 50, 50,
                5, 5, 5, 5, 5, 5]
)

nmpc.quad_stage_cost.add_inputs(
    names=['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'],
    weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

# Horizon
nmpc.horizon = 30

# Box constraints
nmpc.set_box_constraints(
    x_lb=[-10, -10, -10, -10, -10, -10, 
          -10, -10, -10, -10, -10, -10],
    x_ub=[10, 10, 10, 10, 10, 10,
          10, 10, 10, 10, 10, 10],
    u_lb=[-2, -2, -2, -2, -2, -2],
    u_ub=[2, 2, 2, 2, 2, 2],
    z_lb=[0.0]*24,      # <-- enforces obstacle avoidance
    z_ub=[ca.inf]*24
)

# Initial conditions
x0 = [0]*12
z0 = [1.0]*24   # start feasible
u0 = [0]*6

model.set_initial_conditions(x0=x0, z0=z0)
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

nmpc.setup(options={'print_level': 0})


'''
SHELF POSES:
tensor([[-0.7500, -0.6000,  0.2500],
        [-0.7500, -0.3500,  0.0000],
        [-0.7500, -0.6000,  0.7500],
        [-0.7500, -0.6000,  1.2500],
        [-0.7500, -0.1000,  0.2500],
        [-0.7500, -0.1000,  0.7500],
        [-0.7500, -0.1000,  1.2500],
        [-0.7500,  0.4000,  0.2500],
        [-0.7500,  0.4000,  0.7500],
        [-0.7500,  0.4000,  1.2500],
        [-0.7500,  0.9000,  0.2500],
        [-0.7500,  0.9000,  0.7500],
        [-0.7500,  0.9000,  1.2500],
        [-0.7500, -0.3500,  0.5000],
        [-0.7500, -0.3500,  1.0000],
        [-0.7500,  0.1500,  0.0000],
        [-0.7500,  0.1500,  0.5000],
        [-0.7500,  0.1500,  1.0000],
        [-0.7500,  0.6500,  0.0000],
        [-0.7500,  0.6500,  0.5000],
        [-0.7500,  0.6500,  1.0000],
        [-0.7500,  1.1500,  0.0000],
        [-0.7500,  1.1500,  0.5000],
        [-0.7500,  1.1500,  1.0000]], device='cuda:0')

ELLIPSOID RAIUS:
[0.01, 0.01, 0.01]

INITIAL STATE (LIE):
tensor([[-0.9457, -0.2658,  0.9429, -0.1459,  0.0647,  0.3141,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000]], device='cuda:0')

TARGET
tensor([[ 0.0000,  0.0000,  0.0000, -0.3400, -0.1850,  0.3200]],


'''




# # ======================================================
# # Simulation loop
# # ======================================================
# n_steps = 300
# sol = model.solution

# t_dir = "."

# for k in range(n_steps):
#     u_opt = nmpc.optimize(x0)
#     model.simulate(u=u_opt, steps=1)
#     x0 = sol['x:f']
    

#     # Visuals
    
#     fig, axs = plt.subplots(1, 1, figsize=(10,10))
    

#     print(x0, [X_ref, Y_ref], )

#     get_frame(x0, [X_ref, Y_ref], [1, 1], ax=axs)
#     axs.get_xaxis().set_visible(True)
#     axs.get_yaxis().set_visible(True)

#     fig.tight_layout()
#     fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(k)))
#     plt.close(fig)

# print("Simulation finished")
