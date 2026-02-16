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


from rl_manipulation_obstacles.py_dq.src.dq_lie import *
from rl_manipulation_obstacles.py_dq.src.dq import *

from rl_manipulation_obstacles.mpc_controller import *


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

    def plot_ellipsoid(ax, center, radii, color='orange', alpha=1, resolution=20):
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
            shade=True,
            zorder = 6
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
    # Plot obstacles (ellipsoids)
    # =========================
    if obst_centers is not None and obst_radii is not None:
        for center, radii in zip(obst_centers, obst_radii):
            plot_ellipsoid(ax, center, radii)

    # =========================
    # Plot robot
    # =========================
    ax.scatter(X, Y, Z, c='k', s=80, label="Robot", zorder=5)

    # =========================
    # Plot target
    # =========================
    ax.scatter(Xt, Yt, Zt, c='g', s=100, label="Target", zorder=5)


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


# Original pose: -0.6800, -0.3700,  0.7400,  0.2706, -0.6533, -0.2706,  0.6533
ref_lab = torch.tensor([[-0.8800, -0.3645,  0.8800, -0.3400, -0.1850,  0.3700]])
tgt_pos = [-0.6800, -0.3700,  0.7400]

x0 = [-0.9456, -0.2658,  0.9430, -0.1979,  0.0700,  0.3059,  
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]

ellipsoid_r = [0.43/2, 
               0.2/2,  
               0.435/2]

obst_list = torch.tensor(obst_list)

model, nmpc, ellipsoid_r_torch = drop_NMPC_setup(obst_list=obst_list, 
                              ellipsoid_r=ellipsoid_r, 
                              ini = torch.tensor(x0), 
                              ref = ref_lab[0])


# ======================================================
# Simulation loop
# ======================================================
n_steps = 500
sol = model.solution

t_dir = "."

trajectory = []
trajectory_save = []

for k in range(n_steps):
    u_opt = nmpc.optimize(x0)
    model.simulate(u=u_opt, steps=1)
    x0 = sol['x:f']

    x0_tensor = torch.tensor([[float(x0[0]), float(x0[1]), float(x0[2]), 
                        float(x0[3]), float(x0[4]), float(x0[5])]])
    
    x0_group = exp_bruno(x0_tensor)
    x0_lab   = convert_dq_to_Lab(x0_group)

    # print(u_opt)    

    print("Step: ", k)

    trajectory.append([x0_lab[0, 0].item(), x0_lab[0, 1].item(), x0_lab[0, 2].item()])
    trajectory_save.append(x0_tensor[0].numpy().tolist())

    fig, ax = get_frame(
        x0_lab[0],
        tgt=tgt_pos,
        traj=trajectory,
        ax=None,
        obst_centers=obst_list[:, :3],
        obst_radii=ellipsoid_r_torch*2
    )



    # fig.savefig(f"{k:03d}.png")

    ax.view_init(elev=0, azim=0)
    fig.savefig(f"{k:03d}.png")

    # ax.view_init(elev=0, azim=45)
    # fig.savefig(f"side_{k:03d}.png")


    ax.view_init(elev=90, azim=-0)
    fig.savefig(f"other_{k:03d}.png")

    
    plt.close(fig)

    

print("Simulation finished")
save_traj(trajectory_save, lie = True)

# torch.save(torch.tensor(trajectory_save), "./rl_manipulation_obstacles/traj.pt")

