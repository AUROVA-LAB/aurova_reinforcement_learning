# Add HILO-MPC to path. NOT NECESSARY if it was installed via pip.
import sys
sys.path.append('../../../')

from hilo_mpc import NMPC, Model
import casadi as ca

import warnings
warnings.filterwarnings("ignore")

model = Model(plot_backend='bokeh')

# Constants
M = 5.
m = 1.
l = 1.
h = .5
g = 9.81

# States and algebraic variables
x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
model.set_measurement_equations([x[0], x[1], x[2], x[3]])
y = model.set_algebraic_states(['y'])

# Unwrap states
v = x[1]
theta = x[2]
omega = x[3]

# Define inputs
F = model.set_inputs('F')

# ODEs
dd = ca.SX.sym('dd', 4)
dd[0] = v
dd[1] = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
dd[2] = omega
dd[3] = 1. / l * (dd[1] * ca.cos(theta) + g * ca.sin(theta))


# Algebraic equations (note that it is written in the form rhs = 0)
rhs = h + l * ca.cos(theta) - y

# Add differential equations
model.set_dynamical_equations(dd)

# Add algebraic equations
model.set_algebraic_equations(rhs)

# Initial conditions
x0 = [2.5, 0., 0.1, 0.]


# Initial guess algebraic states
z0 = h + l * ca.cos(x0[2]) - h
#Initial guess input
u0 = 0.

# Setup the model
dt = .1
model.setup(dt=dt)

nmpc = NMPC(model)

nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])

nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)

nmpc.horizon = 25

nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])

nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

nmpc.setup(options={'print_level': 0})

n_steps = 100

model.set_initial_conditions(x0=x0, z0=z0)

sol = model.solution

for step in range(n_steps):
    u = nmpc.optimize(x0)
    model.simulate(u=u, steps=1)
    x0 = sol['x:f']
    


print("AAAA")