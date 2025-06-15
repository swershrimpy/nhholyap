import jax
from jax import jit
from functools import partial
import copy
import jax.numpy as jnp
import numpy as np  
from scipy.optimize import minimize
import immrax as irx 
from faulty_nonholonomic_car import FaultyNonHolonomicCar
from interval_functions import overlap_size_lax, propagate_interval_euler

@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6))
def loss_ff(u_ol, x_interval, dt, system, p_nominal, p_actuator_fault, num_steps=10):
    """Loss function: size of the propagated observation interval."""
    x_propagated_1 = copy.deepcopy(x_interval)
    x_propagated_2 = copy.deepcopy(x_interval)
    for _ in range(num_steps):
        x_propagated_1 = propagate_interval_euler(irx.natemb(system), copy.deepcopy(x_propagated_1), u_ol, irx.icentpert(jnp.array([0, 0]), jnp.zeros(2)), p_nominal, dt)
        x_propagated_2 = propagate_interval_euler(irx.natemb(system), copy.deepcopy(x_propagated_2), u_ol, irx.icentpert(jnp.array([0, 0]), jnp.zeros(2)), p_actuator_fault, dt)
    return overlap_size_lax(x_propagated_1[:2], x_propagated_2[:2])  # Only consider position states
# Your objective function (no changes needed here)
def jax_objective_ff(u, x_interval, dt, system, p_nominal, p_actuator_fault, num_steps):
    """A pure JAX function that takes the flat param vector and static args."""
    return loss_ff(u, x_interval, dt, system, p_nominal, p_actuator_fault, num_steps)

#
# >>>>> THIS IS WHERE YOU MAKE THE CHANGE <<<<<
#
# OLD LINE (to be removed):
# value_and_grad_fn = jax.jit(jax.value_and_grad(jax_objective, argnums=0))
#
# NEW, CORRECTED BLOCK:
# Tell JIT to treat arguments 1 through 7 as static constants.
# It will only trace argument 0 ('params').
value_and_grad_fn_ff = jax.jit(
    jax.value_and_grad(jax_objective_ff, argnums=0),
    static_argnums=(1, 2, 3, 4, 5, 6)  # <<< CORRECTED static_argnums
)


# The final wrapper for SciPy now calls this JAX function
def scipy_wrapper_with_grad_ff(params, *args):
    # We must convert JAX arrays to NumPy arrays for SciPy.
    # JAX handles the conversion of numpy inputs automatically.
    value, grad = value_and_grad_fn_ff(params, *args)
    # SciPy optimizers work with float64 by default, so ensure type compatibility.
    # It's safer to return standard numpy arrays.
    return np.float64(value), np.float64(grad) # <<< CHANGED to np.float64


# --- Setup for the optimization ---
def calculate_optimal_open_loop_u(x_interval, p_nominal, p_actuator_fault, observer_offset, dt=1.0, num_steps=10):
    """Calculate the optimal control input.
    
    Args:
        x_interval (irx.Interval): The initial state interval.
        """
    initial_params = np.zeros(2)  # Example: 2D control input
    # Example fixed arguments (assuming your setup)
    # x_interval = irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))
    # dt = .10 # Using a float for dt
    # num_steps = 10
    

    # Assume normal_car_embedding_system is defined
    # from your_file import normal_car_embedding_system

    fixed_args = (x_interval,
                dt,
                FaultyNonHolonomicCar(),
                p_nominal,
                p_actuator_fault,
                num_steps
                )

    print("\nStarting gradient-based optimization with JAX...")
    # Use a method that can leverage gradients, like 'BFGS' or 'L-BFGS-B'
    # Tell SciPy that our function returns the jacobian (gradient) by setting jac=True
    result_grad = minimize(
        scipy_wrapper_with_grad_ff,
        initial_params,
        args=fixed_args,
        method='L-BFGS-B',
        jac=True,  # <<< IMPORTANT! UNCOMMENTED THIS LINE
        options={'disp': True} # <<< UNCOMMENTED to see optimizer progress
    )

    # Extract and display the results
    if result_grad.success:
        print("\nGradient-based optimization successful!")
        optimal_params = result_grad.x
        optimal_u_ol = optimal_params[0:2]
        # optimal_K = optimal_params[2:].reshape((2, 2))
        print(f"Optimal u_ol:\n{optimal_u_ol}")
        # print(f"Optimal K:\n{optimal_K}")
        print(f"Minimum loss value: {result_grad.fun}")
    else:
        print("\nGradient-based optimization failed.")
        print(f"Message: {result_grad.message}")
    return optimal_u_ol, result_grad 

calculate_optimal_open_loop_u(
    x_interval=irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2])),
    p_nominal = irx.icentpert(jnp.array([1.]), jnp.array([0.])), # Assuming valid interval
    p_actuator_fault = irx.icentpert(jnp.array([0.25]), jnp.array([0.25])),
    observer_offset=jnp.ones(4) * 0.,  # Offset for the observer
    dt=.10,
    num_steps=10
)