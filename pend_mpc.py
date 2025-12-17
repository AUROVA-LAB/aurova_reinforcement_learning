import torch

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import pendulum
from robot_model import RobotDx

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm

# matplotlib inline


params = torch.tensor((10., 0.))
dx = RobotDx(params, simple=True)

n_batch, T, mpc_T = 4, 300, 5

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

torch.manual_seed(0)
th = uniform((n_batch, 3), -0.3, 0.3)
thdot = uniform((n_batch, 3), -1., 1.)*0
# print(th)
# print(thdot.shape)
# raise
xinit = torch.cat((th, thdot), dim=1)

x = xinit
u_init = None

# The cost terms for the swingup task can be alternatively obtained
# for this pendulum environment with:
# q, p = dx.get_true_obj()

mode = 'swingup'
# mode = 'spin'

if mode == 'swingup':
    goal_weights = torch.Tensor([1,1,1,
                                0.1, 0.1, 0.1]).repeat(n_batch, 1)
    print("GOAL WEIGHTS: ", goal_weights)
    goal_pos = torch.rand(n_batch, 3)*2 - 1
    print("GOAL POS: ", goal_pos)
    goal_ctrl = torch.zeros_like(goal_pos)
    goal_state = torch.cat((goal_pos, goal_ctrl), dim = -1)
    print("GOAL STATE: ", goal_state)
    ctrl_penalty = 0.001
    print(goal_weights.shape)
    print((ctrl_penalty*torch.ones((n_batch, dx.n_ctrl))).shape)
    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones((n_batch, dx.n_ctrl))
    ), dim = -1)
    print("q: ", q)




    px = -torch.sqrt(goal_weights)*goal_state
    print("PX: ", px)
    p = torch.cat((px, torch.zeros(n_batch, dx.n_ctrl)), dim = -1)
    print("p: ", p.shape)
    Q = torch.diag(q[0]).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )
    print("Q: ", Q.shape)

    p = p.unsqueeze(0).repeat(mpc_T, 1, 1)
    print("p: ", p.shape)
    






t_dir = "."
print('Tmp dir: {}'.format(t_dir))

for t in tqdm(range(T)):
    
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(Q, p), dx)
    
    # print(type(Q.data))
    # print(type(Q))
    # print(p.shape)
    # print(dx)
    
    
    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)

    n_row, n_col = n_batch//2, n_batch//2
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col,3*n_row))
    axs = axs.reshape(-1)
    for i in range(n_batch):
        dx.get_frame(x[i], goal_state[i], ax=axs[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    plt.close(fig)

vid_fname = 'pendulum-{}.mp4'.format(mode)

if os.path.exists(vid_fname):
    os.remove(vid_fname)
    
cmd = (
        '/usr/bin/ffmpeg -y -r 32 -i %03d.png '
        '-vcodec libx264 -crf 25 -pix_fmt yuv420p {}'
    ).format(vid_fname)

os.system(cmd)
print('Saving video to: {}'.format(vid_fname))