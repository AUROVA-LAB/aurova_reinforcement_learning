import sys
sys.path.append('../../../')

from hilo_mpc import NMPC, Model
import casadi as ca
import warnings

import matplotlib.pyplot as plt
import os

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
x = model.set_dynamical_states(['X', 'Y', 'Vx', 'Vy'])
X = x[0]
Y = x[1]
Vx = x[2]
Vy = x[3]

# Measurements
model.set_measurements(['yX', 'yY', 'yVx', 'yVy'])
model.set_measurement_equations(x)

# Inputs
u = model.set_inputs(['ax', 'ay'])
ax = u[0]
ay = u[1]

# ======================================================
# Dynamics
# ======================================================
dx = ca.vertcat(
    Vx,
    Vy,
    ax,
    ay
)
model.set_dynamical_equations(dx)

# ======================================================
# Obstacle avoidance via algebraic constraint
# ======================================================
# Obstacle parameters
Xo = 0.5
Yo = 0.5
r_safe = 0.1

# Algebraic state (constraint slack, optional)
z = model.set_algebraic_states(['c_obs'])

# Constraint equation: c_obs = (X-Xo)^2 + (Y-Yo)^2 - r^2
c_obs = (X - Xo)**2 + (Y - Yo)**2 - r_safe**2
model.set_algebraic_equations(c_obs - z)

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

nmpc.quad_stage_cost.add_states(
    names=['X', 'Y', 'Vx', 'Vy'],
    ref=[X_ref, Y_ref, 0, 0],
    weights=[50, 50, 5, 5]
)

nmpc.quad_stage_cost.add_inputs(
    names=['ax', 'ay'],
    weights=[0.1, 0.1]
)

# Horizon
nmpc.horizon = 30

# Box constraints
nmpc.set_box_constraints(
    x_lb=[-10, -10, -5, -5],
    x_ub=[10, 10, 5, 5],
    u_lb=[-2, -2],
    u_ub=[2, 2],
    z_lb=[0.0],      # <-- enforces obstacle avoidance
    z_ub=[ca.inf]
)

# Initial conditions
x0 = [0, 0, 0, 0]
z0 = [1.0]   # start feasible
u0 = [0, 0]

model.set_initial_conditions(x0=x0, z0=z0)
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

nmpc.setup(options={'print_level': 0})

# ======================================================
# Simulation loop
# ======================================================
n_steps = 300
sol = model.solution

t_dir = "."

for k in range(n_steps):
    u_opt = nmpc.optimize(x0)
    model.simulate(u=u_opt, steps=1)
    x0 = sol['x:f']
    

    # Visuals
    
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    

    print(x0, [X_ref, Y_ref], [Xo, Yo])

    get_frame(x0, [X_ref, Y_ref], [Xo, Yo], ax=axs)
    axs.get_xaxis().set_visible(True)
    axs.get_yaxis().set_visible(True)

    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(k)))
    plt.close(fig)

print("Simulation finished")
