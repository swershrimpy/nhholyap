import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from scipy.optimize import minimize, basinhopping
import copy
import immrax as irx
from interval_functions import overlap_size_lax, propagate_with_feedback, observation_interval_size
from visualization_functions import visualize_trajectory_given_u_K
from faulty_nonholonomic_car import FaultyNonHolonomicCar

normal_car_embedding_system = FaultyNonHolonomicCar()

@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10))
def loss(u_ol, K, x_interval, dt, nominal_system, actuator_fault_system, p_nominal, p_actuator_fault, 
         w_if=irx.icentpert(jnp.zeros(2), jnp.zeros(2)), obs_fault_offset=jnp.ones(4)*0.2, num_steps=10):
    """Loss function: size of the propagated observation interval."""
    x_ivl_nominal = copy.deepcopy(x_interval)
    x_ivl_actuator_fault = copy.deepcopy(x_interval)
    print(num_steps)
    for _ in range(num_steps):
        x_ivl_nominal = propagate_with_feedback(x_ivl_nominal, u_ol, K, dt, nominal_system, w_if=w_if, p_ivl=p_nominal)
        x_ivl_actuator_fault = propagate_with_feedback(x_ivl_actuator_fault, u_ol, K, dt, actuator_fault_system, w_if=w_if, p_ivl=p_actuator_fault)
    return overlap_size_lax(x_ivl_nominal[:2], x_ivl_actuator_fault[:2]) # Only consider position states

@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 10, 11, 12))
def loss_with_weights(u_ol, K, x_interval, dt, nominal_system, actuator_fault_system, p_nominal, p_actuator_fault, 
         w_if=irx.icentpert(jnp.zeros(2), jnp.zeros(2)), obs_fault_offset=jnp.ones(4)*0.2, num_steps=10, learning_rate=0.2, ivl_size_weight=1.0):
    """Loss function: size of the propagated observation interval."""
    x_ivl_nominal = copy.deepcopy(x_interval)
    x_ivl_actuator_fault = copy.deepcopy(x_interval)
    cost = 0.0
    for _ in range(num_steps):
        x_ivl_nominal = propagate_with_feedback(x_ivl_nominal, u_ol, K, dt, nominal_system, w_if=w_if, p_ivl=p_nominal)
        x_ivl_actuator_fault = propagate_with_feedback(x_ivl_actuator_fault, u_ol, K, dt, actuator_fault_system, w_if=w_if, p_ivl=p_actuator_fault)
        x_ivl_observer_fault = copy.deepcopy(x_ivl_actuator_fault) + obs_fault_offset
        this_cost = overlap_size_lax(x_ivl_observer_fault, x_ivl_actuator_fault) + ivl_size_weight * (
            # observation_interval_size(x_ivl_nominal) + 
            observation_interval_size(x_ivl_observer_fault) + 
            observation_interval_size(x_ivl_actuator_fault))
        cost += learning_rate * this_cost # Only consider position states
    return cost

def jax_objective(
        params, 
        x_interval, 
        dt, 
        system1, 
        system2, 
        p_nominal, 
        p_actuator_fault, 
        num_steps,
        ivl_size_weight=1.0,
        learning_rate=0.2,
        obs_offset=jnp.ones(4) * 0.2  # Offset for the observer
    ):
    """A pure JAX function that takes the flat param vector and static args."""
    u_ol = params[0:2] 
    K = params[2:].reshape((2, 2)) 
    return loss_with_weights(
        u_ol=u_ol, 
        K=K,  # Assuming K is zero for the initial guess
        x_interval=x_interval, 
        dt=dt, 
        nominal_system=system1, 
        actuator_fault_system=system2, 
        p_nominal=p_nominal, 
        p_actuator_fault=p_actuator_fault, 
        num_steps=num_steps,
        ivl_size_weight=ivl_size_weight,  # Weight for interval size in the cost function
        learning_rate=learning_rate,  # Fixed disturbance
        obs_fault_offset=obs_offset  # Offset for the observer   
    )

#
# >>>>> THIS IS WHERE YOU MAKE THE CHANGE <<<<<
#
# OLD LINE (to be removed):
# value_and_grad_fn = jax.jit(jax.value_and_grad(jax_objective, argnums=0))
#
# NEW, CORRECTED BLOCK:
# Tell JIT to treat arguments 1 through 7 as static constants.
# It will only trace argument 0 ('params').
value_and_grad_fn = jax.jit(
    jax.value_and_grad(jax_objective, argnums=0),
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9)  # <<< CORRECTED static_argnums
)

value_fn = jax.jit(
    jax_objective, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9)  # <<< CORRECTED static_argnums
)

# The final wrapper for SciPy now calls this JAX function
def scipy_wrapper_with_grad(params, *args):
    # We must convert JAX arrays to NumPy arrays for SciPy.
    # JAX handles the conversion of numpy inputs automatically.
    value, grad = value_and_grad_fn(params, *args)
    # SciPy optimizers work with float64 by default, so ensure type compatibility.
    # It's safer to return standard numpy arrays.
    return np.float64(value), np.float64(grad) # <<< CHANGED to np.float64

def scipy_wrapper_without_grad(params, *args):  
    """Wrapper for SciPy that only returns the function value."""
    # We must convert JAX arrays to NumPy arrays for SciPy.
    # JAX handles the conversion of numpy inputs automatically.
    value = value_fn(params, *args)
    # SciPy optimizers work with float64 by default, so ensure type compatibility.
    return np.float64(value)  # <<< CHANGED to np.float64

def scipy_wrapper_grad_only(params, *args):  
    """Wrapper for SciPy that only returns the function value."""
    # We must convert JAX arrays to NumPy arrays for SciPy.
    # JAX handles the conversion of numpy inputs automatically.
    _, grad = value_and_grad_fn(params, *args)
    # SciPy optimizers work with float64 by default, so ensure type compatibility.
    return np.float64(grad)  # <<< CHANGED to np.float64

# --- Setup for the optimization ---



# Use numpy for the initial parameters that go into the SciPy function
initial_params = np.concatenate([np.zeros(2), np.zeros((2, 2)).flatten()])

# Example fixed arguments (assuming your setup)
x_interval_example = irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))
dt = 0.1 # Using a float for dt
num_steps = 10
p_nominal = irx.icentpert(jnp.array([1.]), jnp.array([0.])) # Assuming valid interval
p_actuator_fault = irx.icentpert(jnp.array([0.25]), jnp.array([0.25]))
observer_offset = jnp.ones(4) * 0.2  # Offset for the observer
learning_rate = 0.2  # Learning rate for the cost function
ivl_size_weight = 1.0  # Weight for interval size in the cost function
# p_no_disturbance = irx.icentpert(jnp.array([1]), jnp.array([.0]))  # No disturbance parameters

# Assume normal_car_embedding_system is defined
# from your_file import normal_car_embedding_system

fixed_args = (x_interval_example,
              dt,
              normal_car_embedding_system,
              normal_car_embedding_system,
              p_nominal,
              p_actuator_fault,
              num_steps,
              ivl_size_weight,
              learning_rate,
              observer_offset  # Offset for the observer
            )
def calculate_optimal_feedback(x_interval, p_nominal, p_actuator_fault, observer_offset=jnp.ones(4)*0.2, dt=1.0, num_steps=10, local_optimization=False,
                               ivl_size_weight=1.0, learning_rate=0.2, disp=False):
    """Calculate the optimal control input."""
    initial_params = np.zeros(6)  # Example: 2D control input + 2x2 gain matrix
    # The 'args' tuple is now part of the instructions for the local minimizer
    fixed_args = (x_interval,
              dt,
              normal_car_embedding_system,
              normal_car_embedding_system,
              p_nominal,
              p_actuator_fault,
              num_steps,
              ivl_size_weight,
              learning_rate,
              observer_offset  # Offset for the observer
            )
    # Use a method that can leverage gradients, like 'BFGS' or 'L-BFGS-B'
    # Tell SciPy that our function returns the jacobian (gradient) by setting jac=True
     # Set to True to use gradient-based optimization
    if local_optimization:
        print("\nStarting gradient-based optimization with JAX...")

        result_grad = minimize(
            scipy_wrapper_without_grad,
            initial_params,
            args=fixed_args,
            method='basinhopping',
            jac=False,  # <<< IMPORTANT! UNCOMMENTED THIS LINE
            options={'disp': disp} # <<< UNCOMMENTED to see optimizer progress
        )

        # Extract and display the results
        if result_grad.success:
            print("\nGradient-based optimization successful!")
            optimal_params = result_grad.x
            optimal_u_ol = optimal_params[0:2]
            optimal_K = optimal_params[2:].reshape((2, 2))
            print(f"Optimal u_ol:\n{optimal_u_ol}")
            print(f"Optimal K:\n{optimal_K}")
            print(f"Minimum loss value: {result_grad.fun}")
        else:
            print("\nGradient-based optimization failed.")
            print(f"Message: {result_grad.message}")

        return optimal_u_ol, optimal_K, result_grad
    else:
        print("\nStarting global optimization with gradients using basinhopping...")
        bounds_list = [(-5.0, 5.0),   # Bound for u_ol[0]
                (-5.0, 5.0),   # Bound for u_ol[1]
                (-2.0, 2.0),   # Bound for K[0,0]
                (-2.0, 2.0),   # Bound for K[0,1]
                (-2.0, 2.0),   # Bound for K[1,0]
                (-2.0, 2.0)]  # Bound for K[1,1]
        # The 'args' tuple is now part of the instructions for the local minimizer
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": scipy_wrapper_grad_only,
            "args": fixed_args,  # <<< CORRECT LOCATION FOR 'args'
            "bounds": bounds_list,  # <<< BOUNDS FOR EACH PARAMETER
            "options": {"disp": disp}  # Display optimization progress
        }

        # The top-level call to basinhopping no longer has the 'args' keyword
        result_nograd = basinhopping(
            scipy_wrapper_without_grad, 
            initial_params,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,  # Or a more appropriate number for your problem
            disp=disp,
        )

        # Extract and display the results
        if result_nograd.lowest_optimization_result.success:
            print("\nGlobal optimization successful!")
            optimal_params = result_nograd.x
            optimal_u_ol = optimal_params[0:2]
            optimal_K = optimal_params[2:].reshape((2, 2))
            print(f"Optimal u_ol:\n{optimal_u_ol}")
            print(f"Optimal K:\n{optimal_K}")
            print(f"Minimum loss value: {result_nograd.fun}")
        else:
            print("\nGlobal optimization failed.")
            print(f"Message: {result_nograd.message}")

        return optimal_u_ol, optimal_K, result_nograd

optimal_u, optimal_K, result_struct = calculate_optimal_feedback(   
    x_interval=x_interval_example,
    dt=dt,
    p_nominal=p_nominal,
    p_actuator_fault=p_actuator_fault,
    num_steps=num_steps,
    ivl_size_weight=ivl_size_weight,
    learning_rate=learning_rate,
    observer_offset=observer_offset  
)  

visualize_trajectory_given_u_K(
    x0_interval=x_interval_example,
    u_ol=optimal_u,
    K=optimal_K,
    dt=dt,
    w_interval=irx.icentpert(jnp.array([0., 0.]), jnp.zeros(2)),  # Assuming no disturbance
    p_no_disturbance=p_nominal,  # Assuming no disturbance parameters
    p_actuator_fault=p_actuator_fault,
    observer_offset=observer_offset,  # Offset for the observer
    max_iter=10,
)