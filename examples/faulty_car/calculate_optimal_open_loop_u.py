import jax
from jax import jit, vmap
from functools import partial
import jax.numpy as jnp
import numpy as np
import immrax as irx
from faulty_nonholonomic_car import FaultyNonHolonomicCar
from interval_functions import overlap_size_lax, propagate_interval_euler, propagate_interval
from visualization_functions import visualize_trajectory_given_u_K
import time
import jaxopt

# --- System and Interval Setup ---
system = FaultyNonHolonomicCar()
natemb_system = irx.natemb(system)
w_ivl = irx.icentpert(jnp.array([0, 0]), jnp.zeros(2))

# --- JAX-compatible Loss Function ---
def loss_ff_jax(u_ol, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps=10):
    """
    Loss function: size of the overlap between nominal and faulty propagated intervals.
    This function is JIT-compatible.
    """
    # Define the single-step propagation for the nominal case
    def nominal_step_fn(i, x_propagated):
        return propagate_interval_euler(natemb_system, x_propagated, u_ol, w_ivl, p_nominal, dt)

    # Define the single-step propagation for the faulty case
    def actuator_fault_step_fn(i, x_propagated):
        return propagate_interval_euler(natemb_system, x_propagated, u_ol, w_ivl, p_actuator_fault, dt)

    # Use fori_loop for efficient JIT-compilation of the propagation loop
    x_ivl_nominal = jax.lax.fori_loop(0, num_steps, nominal_step_fn, x_interval)
    x_ivl_actuator_fault = jax.lax.fori_loop(0, num_steps, actuator_fault_step_fn, x_interval)

    return overlap_size_lax(x_ivl_nominal[:2], x_ivl_actuator_fault[:2])

def loss_ff_jax_multiple(u_ols, x_interval, p_nominal, p_actuator_fault, observer_offset, 
                         dt=0.01, timesteps_per_control=10, num_controls=10):
    """
    Calculates the final overlap size after propagating the system along a
    sequence of control inputs, where each control is held constant for a
    specified number of timesteps.

    Args:
        u_ols (jax.Array): An array of control inputs of shape (N, 2), where N
                           is the number of distinct control decisions.
        x_interval (irx.Interval): The initial state interval.
        dt (float): The time step duration for the simulation.
        p_nominal (irx.Interval): The nominal system parameters.
        p_actuator_fault (irx.Interval): The faulty system parameters.
        timesteps_per_control (int): The number of simulation timesteps to apply
                                     each control input `u_ol`. This must be a
                                     static argument for JIT compilation.

    Returns:
        float: A single scalar representing the final overlap size.
    """
    # The number of control decisions is determined by the length of u_ols.
    # num_controls = u_ols.shape[0]

    # The state we carry through the outer loop is a tuple containing both the
    # nominal and faulty state intervals.
    initial_carry = (x_interval, x_interval)

    def outer_loop_body(i, carry):
        """
        The body of the outer loop. It processes one control input from `u_ols`.

        Args:
            i (int): The current outer loop index (for control inputs).
            carry (tuple): The current state, (x_nominal, x_faulty), from the
                           previous control block.

        Returns:
            tuple: The updated state after applying the current control for
                   `timesteps_per_control` steps.
        """
        # Select the control input for this entire block of timesteps.
        u_current = jax.lax.dynamic_slice(u_ols, (i * 2,), (2,))
        
        # The state for the inner loop starts with the current `carry`.
        inner_initial_carry = carry

        def inner_loop_body(j, inner_carry):
            """
            The body of the inner loop. It simulates one step using the fixed `u_current`.
            """
            x_nom_current, x_fault_current = inner_carry
            
            # Propagate both systems one step forward using the same `u_current`.
            x_nom_next = propagate_interval_euler(natemb_system, x_nom_current, u_current, w_ivl, p_nominal, dt)
            x_fault_next = propagate_interval_euler(natemb_system, x_fault_current, u_current, w_ivl, p_actuator_fault, dt)
            
            return (x_nom_next, x_fault_next)

        # The inner loop applies the fixed `u_current` for `timesteps_per_control` steps.
        final_carry_for_this_control = jax.lax.fori_loop(
            0, timesteps_per_control, inner_loop_body, inner_initial_carry
        )
        
        return final_carry_for_this_control

    # The outer loop iterates through each control decision in `u_ols`.
    final_x_nom, final_x_fault = jax.lax.fori_loop(
        0, num_controls, outer_loop_body, initial_carry
    )
    
    # After all control blocks are processed, calculate the final overlap size.
    return overlap_size_lax(final_x_nom[:2], final_x_fault[:2])

# --- Gradient of the Loss Function ---
# Automatically differentiate the JIT-compiled loss function
loss_grad = jax.grad(loss_ff_jax, argnums=0)
loss_multiple_grad = jax.grad(loss_ff_jax_multiple, argnums=0)
#
# >>>>> THIS IS THE CORRECTED OPTIMIZATION FUNCTION <<<<<
#
def calculate_optimal_open_loop_u(x_interval, p_nominal, p_actuator_fault, observer_offset, dt, num_steps):
    """
    Calculates the optimal open-loop control input `u` using a multi-start gradient descent.
    This entire function is designed to be JIT-compiled.
    """
    solver = jaxopt.LBFGS(
        fun=loss_ff_jax,
        maxiter=50,
        tol=1e-5
    )
    # This inner function performs one run of gradient descent from a random starting point.
    def find_optimal_u_given_initial_guess(prng_key):
        # Hyperparameters for the gradient descent
        learning_rate = 1e1
        num_gd_steps = 10
        
        # subkey, prng_key = jax.random.split(prng_key)
        # Initialize control input `u` with a random guess
        u_initial = jnp.ones(2)

        # # Define the body of the gradient descent loop
        def gradient_step(i, u_current):
            grad = loss_grad(u_current, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps)
            return u_current - learning_rate * grad
        

        # # Use fori_loop for the gradient descent steps. This is crucial for JIT-compilation.
        u_opt = jax.lax.fori_loop(0, num_gd_steps, gradient_step, u_initial)
        # u_opt = solver.run(
        #     init_params=u_initial,
        #     x_interval=x_interval,
        #     dt=dt,
        #     p_nominal=p_nominal,
        #     p_actuator_fault=p_actuator_fault,
        #     observer_offset=observer_offset,
        #     )
    
        return u_opt#u_initial#u_opt

    # --- Main optimization logic ---
    # num_parallel_runs = 1
    # master_key = jax.random.PRNGKey(42)
    # batched_keys = jax.random.split(master_key, num_parallel_runs)

    # # Use vmap to run multiple gradient descents in parallel, each with a different starting key.
    # batch_of_u_opts = vmap(find_optimal_u_given_initial_guess)(batched_keys)

    # # Evaluate the loss for each of the found optima to select the best one.
    # # We vmap the loss function itself.
    # vmapped_loss = vmap(
    #     loss_ff_jax,
    #     # Map over `u_ol` (our batch of optima), keep other args constant.
    #     in_axes=(0, None, None, None, None, None, None)
    # )

    # # CORRECTED: Pass all required arguments to the vmapped loss function.
    # all_losses = vmapped_loss(
    #     batch_of_u_opts, x_interval, dt, p_nominal, p_actuator_fault, observer_offset, num_steps
    # )
    
    # # Find and return the control input that resulted in the minimum loss.
    # best_run_index = jnp.argmin(all_losses)
    return find_optimal_u_given_initial_guess(None)#batch_of_u_opts[best_run_index]

def calculate_optimal_open_loop_u_sequence_gd(x_interval, p_nominal, p_actuator_fault, observer_offset, 
                                           dt=0.01, timesteps_per_control=10, num_controls=10):
    """
    Calculates the optimal open-loop control input `u` using a multi-start gradient descent.
    This entire function is designed to be JIT-compiled.
    """

    learning_rate = 1e1
    num_gd_steps = 100
    u_initial = jnp.ones(2 * num_controls)  # Initialize control input `u` with a random guess
    def gradient_step(i, u_current):
        grad = loss_multiple_grad(
            u_current, 
            x_interval, 
            p_nominal, 
            p_actuator_fault, 
            observer_offset,
            dt,
            timesteps_per_control,
            num_controls
        ) 
        return u_current - learning_rate * grad
    u_opt = jax.lax.fori_loop(0, num_gd_steps, gradient_step, u_initial)
    return u_opt

# --- Setup for the optimization ---
x_interval_example = irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.2, 0.2, 0.2, 0.2]))
dt = 0.1
dt_seq = 0.01
num_steps = 10
p_nominal = irx.icentpert(jnp.array([1.]), jnp.array([0.]))
p_actuator_fault = irx.icentpert(jnp.array([0.25]), jnp.array([0.25]))
observer_offset = jnp.ones(4) * 0.2

# --- JIT-compiling the main function ---
# Use partial to fix the parameters that don't change per run.
# The JIT compilation will happen on the first call.
jit_calculate_optimal_open_loop_u = jit(partial(
    calculate_optimal_open_loop_u,
    p_nominal=p_nominal,
    p_actuator_fault=p_actuator_fault,
    observer_offset=observer_offset,
    dt=dt,
    num_steps=num_steps
))

jit_calculate_optimal_open_loop_u_sequence = jit(partial(
    calculate_optimal_open_loop_u_sequence_gd,
    p_nominal=p_nominal,
    p_actuator_fault=p_actuator_fault,
    observer_offset=observer_offset,
    dt=dt_seq,
    timesteps_per_control=num_steps,
    num_controls=num_steps,
))
SEQUENCE = False
if SEQUENCE:    
    print("Starting JIT compilation (first run)...")
    t0 = time.time()
    # First call triggers compilation. The argument is the initial state `x_interval`.
    optimal_u_precompiled = jit_calculate_optimal_open_loop_u_sequence(x_interval_example)
    jax.block_until_ready(optimal_u_precompiled) # Wait for compilation to finish
    t1 = time.time()
    print(f"Time taken for compilation: {t1 - t0:.6f} seconds")

    print("\nRunning optimized function...")
    t0 = time.time()
    # Subsequent calls use the cached, compiled function and are much faster.
    optimal_u = jit_calculate_optimal_open_loop_u_sequence(x_interval_example)
    # optimal_u = calculate_optimal_open_loop_u_sequence_gd(
    #     x_interval=x_interval_example,
    #     p_nominal=p_nominal,
    #     p_actuator_fault=p_actuator_fault,
    #     observer_offset=observer_offset,
    #     dt=dt_seq,
    #     timesteps_per_control=num_steps,
    #     num_controls=num_steps
    # )
    jax.block_until_ready(optimal_u)
    t1 = time.time()
    print(f"Time taken for optimization: {t1 - t0:.6f} seconds")
    print(f"Optimal open-loop control input: {optimal_u}")

    # Calculate the final loss value with the optimal control
    optimal_loss = loss_ff_jax(
        optimal_u,
        x_interval_example,
        dt,
        p_nominal,
        p_actuator_fault,
        observer_offset,
        num_steps
    )
    print(f"Optimal loss value: {optimal_loss}")
else:
    print("Starting JIT compilation (first run)...")
    t0 = time.time()
    # First call triggers compilation. The argument is the initial state `x_interval`.
    optimal_u_precompiled = jit_calculate_optimal_open_loop_u(x_interval_example)
    jax.block_until_ready(optimal_u_precompiled) # Wait for compilation to finish
    t1 = time.time()
    print(f"Time taken for compilation: {t1 - t0:.6f} seconds")

    print("\nRunning optimized function...")
    t0 = time.time()
    # Subsequent calls use the cached, compiled function and are much faster.
    optimal_u = jit_calculate_optimal_open_loop_u(x_interval_example)
    # optimal_u = calculate_optimal_open_loop_u(
    #     x_interval=x_interval_example,
    #     p_nominal=p_nominal,
    #     p_actuator_fault=p_actuator_fault,
    #     observer_offset=observer_offset,
    #     dt=dt,
    #     num_steps=num_steps
    # )
    jax.block_until_ready(optimal_u)
    t1 = time.time()
    print(f"Time taken for optimization: {t1 - t0:.6f} seconds")
    print(f"Optimal open-loop control input: {optimal_u}")

    # Calculate the final loss value with the optimal control
    optimal_loss = loss_ff_jax(
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
    VISUALIZE = True
    if VISUALIZE:
        visualize_trajectory_given_u_K(
            x0_interval=x_interval_example,
            u_ol=optimal_u,
            K=jnp.zeros((2, 2)),
            dt=0.02,
            w_interval=irx.icentpert(jnp.array([0., 0.]), jnp.zeros(2)),
            p_no_disturbance=p_nominal,
            p_actuator_fault=p_actuator_fault,
            observer_offset=observer_offset,
            max_iter=50,
        )