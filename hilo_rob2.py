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


from rl_manipulation_obstacles.py_dq.src.dq_lie import *
from rl_manipulation_obstacles.py_dq.src.dq import *



def get_frame(x, tgt, obst_centers=None, obst_radii=None, traj=None, ax=None):
    """
    x     : current state (at least X,Y,Z)
    tgt   : [Xt, Yt, Zt]
    obst_centers : list or array [[xo, yo, zo], ...]
    obst_radii   : list or array [[rx, ry, rz], ...]
    traj  : list or array of past positions [[x,y,z], ...]
    ax    : matplotlib 3D axis
    """

    import numpy as np
    import matplotlib.pyplot as plt

    def plot_ellipsoid(ax, center, radii, color='orange', alpha=0.25, resolution=20):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        ax.plot_surface(
            x, y, z,
            color=color,
            alpha=alpha,
            linewidth=0,
            shade=True
        )

    # =========================
    # Extract positions
    # =========================
    X, Y, Z = float(x[0]), float(x[1]), float(x[2])
    Xt, Yt, Zt = tgt

    # =========================
    # Create / clear axis
    # =========================
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
        ax.cla()

    # =========================
    # Plot trajectory
    # =========================
    if traj is not None and len(traj) > 1:
        traj = np.asarray(traj)
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            'k-', linewidth=2, alpha=0.7, label="Trajectory"
        )

    # =========================
    # Plot robot
    # =========================
    ax.scatter(X, Y, Z, c='k', s=80, label="Robot", zorder=5)

    # =========================
    # Plot target
    # =========================
    ax.scatter(Xt, Yt, Zt, c='g', s=100, label="Target", zorder=5)

    # =========================
    # Plot obstacles (ellipsoids)
    # =========================
    if obst_centers is not None and obst_radii is not None:
        for center, radii in zip(obst_centers, obst_radii):
            plot_ellipsoid(ax, center, radii)

    # =========================
    # Plot origin
    # =========================
    ax.scatter(0, 0, 0, c='r', s=80, label="Origin")

    # =========================
    # Axis limits
    # =========================
    xs = [X, Xt, 0]
    ys = [Y, Yt, 0]
    zs = [Z, Zt, 0]

    if obst_centers is not None:
        for c in obst_centers:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])

    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_zlim(min(zs) - margin, max(zs) + margin)

    # =========================
    # Formatting
    # =========================
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Robot Trajectory with Obstacles")
    ax.legend(loc="upper left")



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

obst_list = []

for i in range(2):
    if i == 0:
        obst_list.append(p1)
        p_ = copy.deepcopy(p1)
    else:
        obst_list.append(p2)
        p_ = copy.deepcopy(p2)

    for j in range(4):

        if j != 0:
            p_[1] += 0.5
            obst_list.append(copy.deepcopy(p_))
        
        p__ = copy.deepcopy(p_)

        for k in range(3):
            p__[2] += 0.5
            obst_list.append(copy.deepcopy(p__))


z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(obst_list))])

obst_list = torch.tensor(obst_list)

rhs = []


n_obst = len(obst_list)

# obst_list_group = dq_from_tr(obst_list[:, :3], obst_list[:, 3:])
# obst_list_lie = log_bruno(obst_list_group)


idx_obst = [[0, 1, 2], [2, 0, 1]]
ellipsoid_r = [0.4275 , 
               0.18 , 
               0.42  ]
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

    rhs.append((X - o[0].item())**2 / ellipsoid_r[idx_obst[change_idx][0]]**2 + \
               (Y - o[1].item())**2 / ellipsoid_r[idx_obst[change_idx][1]]**2 + \
               (Z - o[2].item())**2 / ellipsoid_r[idx_obst[change_idx][2]]**2 - 1 - z[idx])
    
    ellipsoid_r_torch.append([ellipsoid_r[idx_obst[change_idx][0]], ellipsoid_r[idx_obst[change_idx][1]], ellipsoid_r[idx_obst[change_idx][2]]])

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

# EULER: -0.2968, -0.0151,  0.4611, -3.0582,  0.9217,  2.6561
# LIE: 0.8948  -0.3471  0.8949  -0.34   0.0687   0.3558

ref_lab = torch.tensor([[-0.6800, -0.3700, 0.75400, -3.0582, 0.9217, 2.6561]])

# Target
X_ref = ref_lab[0, 0].item()
Y_ref = ref_lab[0, 1].item()
Z_ref = ref_lab[0, 2].item()
X__ref = ref_lab[0, 3].item()
Y__ref = ref_lab[0, 4].item()
Z__ref = ref_lab[0, 5].item()

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
    z_lb=[0.0]*n_obst,      # <-- enforces obstacle avoidance
    z_ub=[ca.inf]*n_obst
)

# Initial conditions
x0 = [-0.2018,  0.1293,  0.6283, -3.0582,  0.9217,  2.6561,  
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]
z0 = [1.0]*n_obst   # start feasible
u0 = [0]*6

model.set_initial_conditions(x0=x0, z0=z0)
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

nmpc.setup(options={'print_level': 0})




# ======================================================
# Simulation loop
# ======================================================
n_steps = 300
sol = model.solution

t_dir = "."

trajectory = []

for k in range(n_steps):
    u_opt = nmpc.optimize(x0)
    model.simulate(u=u_opt, steps=1)
    x0 = sol['x:f']

    x0_tensor = torch.tensor([[float(x0[0]), float(x0[1]), float(x0[2]), 
                        float(x0[3]), float(x0[4]), float(x0[5])]])
    
    # x0_group = exp_bruno(x0_tensor)
    # x0_lab = convert_dq_to_Lab(x0_group)

    # print(u_opt)    

    trajectory.append([x0_tensor[0, 0].item(), x0_tensor[0, 1].item(), x0_tensor[0, 2].item()])

    fig, ax = get_frame(
        x0_tensor[0],
        tgt=[ref_lab[0,0].item(), ref_lab[0,1].item(), ref_lab[0,2].item()],
        traj=trajectory,
        ax=None,
        obst_centers=obst_list[:, :3],
        obst_radii=ellipsoid_r_torch
    )

    


    # fig.savefig(f"{k:03d}.png")

    ax.view_init(elev=0, azim=0)
    fig.savefig(f"{k:03d}.png")

    # ax.view_init(elev=0, azim=45)
    # fig.savefig(f"side_{k:03d}.png")


    ax.view_init(elev=90, azim=-0)
    fig.savefig(f"other_{k:03d}.png")

    
    plt.close(fig)

    # Visuals
    
    # fig, axs = plt.subplots(1, 1, figsize=(10,10))
    

    # print(x0, [X_ref, Y_ref], )

    # get_frame(x0, [X_ref, Y_ref], [1, 1], ax=axs)
    # axs.get_xaxis().set_visible(True)
    # axs.get_yaxis().set_visible(True)

    # fig.tight_layout()
    # fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(k)))
    # plt.close(fig)

print("Simulation finished")
