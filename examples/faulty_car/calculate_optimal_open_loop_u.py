import jax
from jax import jit
from functools import partial
import copy
import jax.numpy as jnp
import numpy as np  
from jaxopt import LBFGSB
import immrax as irx 
from faulty_nonholonomic_car import FaultyNonHolonomicCar
from interval_functions import overlap_size_lax, propagate_interval_euler
from visualization_functions import visualize_trajectory_given_u_K
import timeit, time
import jaxopt, varipeps
system = FaultyNonHolonomicCar()
natemb_system = irx.natemb(system)
w_ivl = irx.icentpert(jnp.array([0, 0]), jnp.zeros(2))
# 
def loss_ff(u_ol, x_interval, dt, p_nominal, p_actuator_fault, num_steps=10):
    """Loss function: size of the propagated observation interval."""
    x_propagated_1 = x_interval
    x_propagated_2 = x_interval
    for _ in range(num_steps):
        x_propagated_1 = propagate_interval_euler(natemb_system, x_propagated_1, u_ol, w_ivl, p_nominal, dt)
        x_propagated_2 = propagate_interval_euler(natemb_system, x_propagated_2, u_ol, w_ivl, p_actuator_fault, dt)
    return overlap_size_lax(x_propagated_1[:2], x_propagated_2[:2])  # Only consider position states
# Your objective function (no changes needed here)
def jax_objective_ff(u, x_interval, dt, p_nominal, p_actuator_fault, num_steps):
    """A pure JAX function that takes the flat param vector and static args."""
    return loss_ff(u, x_interval, dt, p_nominal, p_actuator_fault, num_steps)

#
# >>>>> THIS IS WHERE YOU MAKE THE CHANGE <<<<<
#
# OLD LINE (to be removed):
# value_and_grad_fn = jax.jit(jax.value_and_grad(jax_objective, argnums=0))
#
# NEW, CORRECTED BLOCK:
# Tell JIT to treat arguments 1 through 7 as static constants.
# It will only trace argument 0 ('params').
value_and_grad_fn_ff = jax.value_and_grad(jax_objective_ff, argnums=0)           #jax.jit(
    
#     static_argnums=(2, 3, 4, 5)  # <<< CORRECTED static_argnums
# )


# The final wrapper for SciPy now calls this JAX function
def scipy_wrapper_with_grad_ff(params, *args):
    # We must convert JAX arrays to NumPy arrays for SciPy.
    # JAX handles the conversion of numpy inputs automatically.
    value, grad = value_and_grad_fn_ff(params, *args)
    # SciPy optimizers work with float64 by default, so ensure type compatibility.
    # It's safer to return standard numpy arrays.
    return value, grad # <<< CHANGED to np.float64


jaxscipy = jaxopt.ScipyMinimize(
    fun=scipy_wrapper_with_grad_ff,
    method='L-BFGS-B',
    maxiter=100,
    tol=1e-6,
    options={'disp': True}
)

loss_grad = jax.grad(loss_ff, argnums=0)

# --- Setup for the optimization ---

def calculate_optimal_open_loop_u(x_interval, p_nominal, p_actuator_fault, observer_offset, dt=1.0, num_steps=10):
    """Calculate the optimal control input.
    
    Args:
        x_interval (irx.Interval): The initial state interval.
        """
    print("\nStarting gradient-based optimization with JAX...")


    # lr = 1e-1
    # steps = 10
    u_opt = jnp.zeros(2)  # Initial guess for the open-loop control input
    loss_lam = lambda u: loss_ff(u, x_interval, dt, p_nominal, p_actuator_fault, num_steps)
    lbfgsb_solver = jaxopt.LBFGSB(
        fun=loss_lam,
        verbose=True,
        # static_argnums=(2, 3, 4, 5)
    )
    u_opt = lbfgsb_solver.run(
        u_opt, 
        bounds=(jnp.array([-1., -1.]), jnp.array([1., 1.])),  # Assuming control input bounds
        ).params

    # for i in range(steps):
    #     grad = loss_grad(u_opt, x_interval, dt, p_nominal, p_actuator_fault, num_steps)
    #     u_opt -= lr * grad


    return u_opt

jit_calculate_optimal_open_loop_u = jit(partial(calculate_optimal_open_loop_u, 
    # x_interval=irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2])),
    p_nominal = irx.icentpert(jnp.array([1.]), jnp.array([0.])), # Assuming valid interval
    p_actuator_fault = irx.icentpert(jnp.array([0.25]), jnp.array([0.25])),
    observer_offset=jnp.ones(4) * 0.2,  # Offset for the observer
    dt=.10,
    num_steps=3
))

t0 = time.time()
jax.block_until_ready(jit_calculate_optimal_open_loop_u(irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))))
t1 = time.time()
print(f"Time taken for compilation: {t1 - t0:.6f} seconds")

t0 = time.time()
optimal_u = jax.block_until_ready(jit_calculate_optimal_open_loop_u(irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))))
t1 = time.time()
print(f"Time taken for optimization: {t1 - t0:.6f} seconds")

# A running example.
# Example fixed arguments (assuming your setup)
x_interval_example = irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))
dt = 0.1 # Using a float for dt
num_steps = 10
p_nominal = irx.icentpert(jnp.array([1.]), jnp.array([0.])) # Assuming valid interval
p_actuator_fault = irx.icentpert(jnp.array([0.25]), jnp.array([0.25]))
observer_offset = jnp.ones(4) * 0.2  # Offset for the observer
learning_rate = 0.2  # Learning rate for the cost function
ivl_size_weight = 1.0  # Weight for interval size in the cost function
VISUALIZE = False
if VISUALIZE:
    visualize_trajectory_given_u_K(
        x0_interval=x_interval_example,
        u_ol=optimal_u,
        K=jnp.zeros((2, 2)),  # no feedback control for this example
        dt=dt,
        w_interval=irx.icentpert(jnp.array([0., 0.]), jnp.zeros(2)),  # Assuming no disturbance
        p_no_disturbance=p_nominal,  # Assuming no disturbance parameters
        p_actuator_fault=p_actuator_fault,
        observer_offset=observer_offset,  # Offset for the observer
        max_iter=10,
    )