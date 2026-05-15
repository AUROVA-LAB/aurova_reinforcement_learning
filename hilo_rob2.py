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

from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

from scipy.spatial.transform import Rotation
from math import atan, atan2, pi



def get_frame(x, tgt, obst_centers=None, obst_radii=None, traj=None, ax=None, rot = False):
    """
    x     : current state (at least X,Y,Z)
    tgt   : [Xt, Yt, Zt]
    obst_centers : list or array [[xo, yo, zo], ...]
    obst_radii   : list or array [[rx, ry, rz], ...]
    traj  : list or array of past positions [[x,y,z], ...]
    ax    : matplotlib 3D axis
    """

    

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
    X, Y, Z = float(x[0 + 3*int(rot)]), float(x[1 + 3*int(rot)]), float(x[2 + 3*int(rot)])
    Xt, Yt, Zt = float(tgt[0 + 3*int(rot)]), float(tgt[1 + 3*int(rot)]), float(tgt[2 + 3*int(rot)])

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
            traj[:, 0 + 3*int(rot)], traj[:, 1 + 3*int(rot)], traj[:, 2 + 3*int(rot)],
            'k-', linewidth=2, alpha=0.7, label="Trajectory"
        )
    # =========================
    # Plot obstacles (ellipsoids)
    # =========================
    if obst_centers.cpu().numpy().tolist()[0] != [] and obst_radii.cpu().numpy().tolist()[0] != []:
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


    if obst_centers.cpu().numpy().tolist()[0] != []:
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



references = [torch.tensor([[ 0.9753,  1.2177,  0.0294, -0.2032, -0.2433,  0.1176]], device='cuda:0'), 
              torch.tensor([[ 0.9753,  1.2177,  0.0294, -0.2032, -0.2433,  0.0051]], device='cuda:0'), 
              torch.tensor([[ 0.0000,  1.5000,  0.0000, -0.3265,  0.2903,  0.15600]], device='cuda:0')]


x0 = [0.08660679310560226, -1.157797932624817, -0.3822425305843353, -0.2346031814813614, 0.03121274895966053, 0.12371678650379181, 
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

obst_list = torch.tensor([[-0.3810,  0.0667,  0.0875]], device='cuda:0')
ellipsoid_r = [0.16, 0.16, 0.16]
dt = 0.1
n_steps_mpc = 200
path_traj_mpc = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation_obstacles/trajectories"

save_idx = 0
trajectory_save = []

for idx, ref in enumerate(references):

    model, nmpc, ellipsoid_r_torch = drop_NMPC_setup(obst_list, 
                                            ellipsoid_r, 
                                            ini = x0, 
                                            ref = ref[0],
                                            dt = dt, 
                                            lie = True,)

    # ======================================================
    # Simulation loop
    # ======================================================
    
    sol = model.solution


    # x0 = x0[0].cpu().numpy().tolist()

    for k in range(n_steps_mpc):
        u_opt = nmpc.optimize(x0)
        model.simulate(u=u_opt, steps=1)
        x0 = sol['x:f']

        x0_tensor = torch.tensor([[float(x0[0]), float(x0[1]), float(x0[2]), 
                                float(x0[3]), float(x0[4]), float(x0[5])]])
        
        x0_group = exp_bruno(x0_tensor)
        x0_lab   = convert_dq_to_Lab(x0_group)

        trajectory_save.append(x0_tensor[0].numpy().tolist())

        if True:
            fig, ax = get_frame(
                x0_tensor[0].cpu(),
                tgt=ref[0].cpu().numpy().tolist(),
                traj=trajectory_save,
                ax=None,
                obst_centers=obst_list.cpu(),
                obst_radii=ellipsoid_r_torch.cpu(),
                rot = True,)
    
            name = f"{save_idx:03d}.png"
            # fig.savefig(os.path.join(path, name))

            name = f"{save_idx:03d}.png"
            ax.view_init(elev=0, azim=0)
            fig.savefig(os.path.join(path_traj_mpc, name))


            f"side_{save_idx:03d}.png"
            # ax.view_init(elev=0, azim=45)
            # fig.savefig(os.path.join(path, name))


            name = f"other_{save_idx:03d}.png"
            ax.view_init(elev=90, azim=-0)
            fig.savefig(os.path.join(path_traj_mpc, name))
            
            plt.close(fig)


        save_idx += 1
        
        if idx == 1:
            start_grip_idx = save_idx

        if torch.norm(x0_tensor.to("cuda:0") - ref).item() < 0.08:
            break


# torch.save(torch.tensor(trajectory_save), "./rl_manipulation_obstacles/traj.pt")

