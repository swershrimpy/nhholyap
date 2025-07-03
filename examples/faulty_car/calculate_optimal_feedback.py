import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
from scipy.optimize import minimize, basinhopping
import copy, time
import immrax as irx
from interval_functions import overlap_size_lax, propagate_with_feedback, compute_traj_with_feedback
from visualization_functions import visualize_trajectory_given_u_K
from faulty_nonholonomic_car import FaultyNonHolonomicCar

normal_car_embedding_system = irx.natemb(FaultyNonHolonomicCar())
w_if=irx.icentpert(jnp.zeros(2), jnp.zeros(2))
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


# --- JAX-compatible Loss Function ---
def loss_fb_jax(u_ol, K, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps=10):
    """
    Loss function: size of the overlap between nominal and faulty propagated intervals.
    This function is JIT-compatible.
    """
    # Define the single-step propagation for the nominal case
    def nominal_step_fn(i, x_propagated):
        return propagate_with_feedback(x_propagated, u_ol, K, dt, normal_car_embedding_system, w_if=w_if, p_ivl=p_nominal)

    # Define the single-step propagation for the faulty case
    def actuator_fault_step_fn(i, x_propagated):
        return propagate_with_feedback(x_propagated, u_ol, K, dt, normal_car_embedding_system, w_if=w_if, p_ivl=p_actuator_fault)
    # Use fori_loop for efficient JIT-compilation of the propagation loop
    x_ivl_nominal = jax.lax.fori_loop(0, num_steps, nominal_step_fn, x_interval)
    x_ivl_actuator_fault = jax.lax.fori_loop(0, num_steps, actuator_fault_step_fn, x_interval)

    return overlap_size_lax(x_ivl_nominal[:2], x_ivl_actuator_fault[:2])

def loss_fb_parallel(params: jnp.ndarray, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps=10):
    u_ol = params[:2]
    K = params[2:].reshape((2, 2))
    """Loss function for parallel optimization: size of the propagated observation interval."""
    return loss_fb_jax(u_ol, jnp.zeros((2,2)), x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps)
# --- Setup for the optimization ---

loss_fb_grad = jax.grad(loss_fb_parallel, argnums=0)  # Gradient of the loss function w.r.t. the parameters


# Use numpy for the initial parameters that go into the SciPy function
initial_params = np.concatenate([jnp.zeros(2), jnp.zeros((2, 2)).flatten()])

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
def calculate_optimal_feedback(x_interval, p_nominal, p_actuator_fault, observer_offset=jnp.ones(4)*0.2, dt=1.0, num_steps=10,
                               ivl_size_weight=1.0, learning_rate=1e1, num_gd_steps=10):
    """Calculate the optimal control input."""  
    # def find_optimal_u_given_initial_guess(prng_key):
    #     # Hyperparameters for the gradient descent
    #     subkey, prng_key = jax.random.split(prng_key)
        # Initialize control input `u` with a random guess
    # u_initial = jax.random.uniform(subkey, shape=(2,), minval=-2.0, maxval=2.0)
    # K_initial = jax.random.uniform(subkey, shape=(2, 2), minval=-1.0, maxval=1.0)
    u_initial = jnp.zeros(2)  # Initial guess for the control input
    K_initial = jnp.zeros((2, 2))  # Initial guess for the
    initial_params = jnp.concatenate([u_initial, K_initial.flatten()])  # Combine u and K into a single parameter vector
    loss_fb_parallel(params=initial_params,
                    x_interval=x_interval,
                    dt=dt,
                    p_nominal=p_nominal,
                    p_actuator_fault=p_actuator_fault,  
                    observer_offset=observer_offset,
                    num_steps=num_steps)
    # Define the body of the gradient descent loop

    def gradient_step(i, u_current):
        grad = loss_fb_grad(u_current, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps)
        return u_current - learning_rate * grad
    param = u_initial#jnp.concatenate([u_initial, K_initial.flatten()])

    # # Use fori_loop for the gradient descent steps. This is crucial for JIT-compilation.
    u_opt = jax.lax.fori_loop(0, num_gd_steps, gradient_step, initial_params)
    return u_opt#u_opt[:2], u_opt[2:].reshape((2,2))#u_opt
    
    # --- Main optimization logic ---
    # num_parallel_runs = 1000
    # master_key = jax.random.PRNGKey(42)
    # batched_keys = jax.random.split(master_key, num_parallel_runs)

    # # Use vmap to run multiple gradient descents in parallel, each with a different starting key.
    # batch_of_u_opts = vmap(find_optimal_u_given_initial_guess)(batched_keys)

    # # Evaluate the loss for each of the found optima to select the best one.
    # # We vmap the loss function itself.
    # vmapped_loss = vmap(
    #     loss_fb_parallel,
    #     # Map over `u_ol` (our batch of optima), keep other args constant.
    #     in_axes=(0, None, None, None, None, None, None)
    # )

    # # CORRECTED: Pass all required arguments to the vmapped loss function.
    # all_losses = vmapped_loss(
    #     batch_of_u_opts, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps
    # )
    
    # # Find and return the control input that resulted in the minimum loss.
    # best_run_index = jnp.argmin(all_losses)
    # return batch_of_u_opts[best_run_index]


jit_calculate_optimal_feedback = jit(partial(
    calculate_optimal_feedback,
    p_nominal=p_nominal,
    p_actuator_fault=p_actuator_fault,
    observer_offset=observer_offset,
    dt=dt,
    num_steps=num_steps,
    ivl_size_weight=ivl_size_weight,
    learning_rate=learning_rate,
))

# optimal_u, optimal_K = calculate_optimal_feedback(   
#     x_interval=x_interval_example,
#     dt=dt,
#     p_nominal=p_nominal,
#     p_actuator_fault=p_actuator_fault,
#     num_steps=num_steps,
#     ivl_size_weight=ivl_size_weight,
#     learning_rate=learning_rate,
#     observer_offset=observer_offset  
# )  

print("Starting JIT compilation (first run)...")
t0 = time.time()
# First call triggers compilation. The argument is the initial state `x_interval`.
optimal_u_precompiled = jit_calculate_optimal_feedback(x_interval_example)
jax.block_until_ready(optimal_u_precompiled) # Wait for compilation to finish
t1 = time.time()
print(f"Time taken for compilation: {t1 - t0:.6f} seconds")

print("\nRunning optimized function...")
t0 = time.time()
# Subsequent calls use the cached, compiled function and are much faster.
# optimal_u = jit_calculate_optimal_feedback(x_interval_example)
optimal_u = calculate_optimal_feedback(
    x_interval=x_interval_example,
    p_nominal=p_nominal,
    p_actuator_fault=p_actuator_fault,
    observer_offset=observer_offset,
    dt=dt,
    num_steps=num_steps,
    ivl_size_weight=ivl_size_weight,
    learning_rate=learning_rate
)
# jax.block_until_ready(optimal_u)
t1 = time.time()
print(f"Time taken for optimization: {t1 - t0:.6f} seconds")
print(f"Optimal open-loop control input: {optimal_u[:2]}, Optimal feedback gain K: {optimal_u[2:].reshape((2, 2))}")

# Calculate the final loss value with the optimal control
optimal_loss = loss_fb_parallel(
    optimal_u,
    x_interval_example,
    dt,
    p_nominal,
    p_actuator_fault,
    observer_offset,
    num_steps
)
print(f"Optimal loss value: {optimal_loss}")

# --- Visualization (Optional) ---
VISUALIZE = False
if VISUALIZE:
    visualize_trajectory_given_u_K(
        x0_interval=x_interval_example,
        u_ol=optimal_u,
        K=jnp.zeros((2, 2)),
        dt=dt,
        w_interval=irx.icentpert(jnp.array([0., 0.]), jnp.zeros(2)),
        p_no_disturbance=p_nominal,
        p_actuator_fault=p_actuator_fault,
        observer_offset=observer_offset,
        max_iter=10,
    )