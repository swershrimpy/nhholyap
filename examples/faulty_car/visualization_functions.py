import immrax as irx
import jax
import jax.numpy as jnp
from jax import random
from faulty_nonholonomic_car import FaultyNonHolonomicCar
from typing import Union
from immutabledict import immutabledict
import matplotlib.pyplot as plt
from interval_functions import overlap_size, overlap_size_lax


rng_key = random.PRNGKey(42)
real_car_system = FaultyNonHolonomicCar()

def propagate_interval(
        embedding_system: irx.System, 
        x0_interval: irx.Interval, 
        u_interval: Union[jnp.ndarray, irx.Interval], 
        w_interval: Union[jnp.ndarray, irx.Interval], 
        p_interval: Union[jnp.ndarray, irx.Interval], 
        t_end: float
    ) -> irx.Interval:
    def u_map_in_func(t, x):
        return u_interval
    def w_map_in_func(t, x):
        return w_interval
    def p_map_in_func(t, x):
        return p_interval
    x_traj = embedding_system.compute_trajectory(0.0, t_end, irx.i2ut(x0_interval), (u_map_in_func, w_map_in_func, p_map_in_func))
    x_emb = x_traj.ys[-1]
    x_ivl = irx.Interval(lower=x_emb[:4], upper=x_emb[4:])
    return x_ivl

def propagate_interval_euler(
        embedding_system: irx.System, 
        x0_interval: irx.Interval, 
        u_interval: Union[jnp.ndarray, irx.Interval], 
        w_interval: Union[jnp.ndarray, irx.Interval], 
        p_interval: Union[jnp.ndarray, irx.Interval], 
        t_end: float
    ) -> irx.Interval:
    xt_ut =  embedding_system.f(0., irx.i2ut(x0_interval), u_interval, w_interval, p_interval) * t_end + irx.i2ut(x0_interval)
    return irx.ut2i(xt_ut)

def plot_observation_boxes(x_interval_actuator_fault, x_interval_no_fault, x_interval_observation_fault, starting_observation, new_observation_interval=None, new_observation=None):
    # Extract the x and y positions from the final interval
    x_pos_interval_actuator_fault = x_interval_actuator_fault[0]  # x position interval
    y_pos_interval_actuator_fault = x_interval_actuator_fault[1]  # y position interval
    x_pos_interval_no_fault = x_interval_no_fault[0]
    y_pos_interval_no_fault = x_interval_no_fault[1]
    x_pos_interval_observation_fault = x_interval_observation_fault[0]
    y_pos_interval_observation_fault = x_interval_observation_fault[1]

    # Plot the interval of x and y positions
    plt.figure(figsize=(8, 6))
    plt.fill_between(
        [x_pos_interval_actuator_fault.lower, x_pos_interval_actuator_fault.upper],
        [y_pos_interval_actuator_fault.lower, y_pos_interval_actuator_fault.lower],
        [y_pos_interval_actuator_fault.upper, y_pos_interval_actuator_fault.upper],
        alpha=0.5,
        color="blue",
        label="Observation Set (x, y) - Actuator Fault"
    )
    plt.fill_between(
        [x_pos_interval_no_fault.lower, x_pos_interval_no_fault.upper],
        [y_pos_interval_no_fault.lower, y_pos_interval_no_fault.lower],
        [y_pos_interval_no_fault.upper, y_pos_interval_no_fault.upper],
        alpha=0.5,
        color="green",
        label="Observation Set (x, y) - No Fault"
    )
    plt.fill_between(
        [x_pos_interval_observation_fault.lower, x_pos_interval_observation_fault.upper],
        [y_pos_interval_observation_fault.lower, y_pos_interval_observation_fault.lower],
        [y_pos_interval_observation_fault.upper, y_pos_interval_observation_fault.upper],
        alpha=0.5,
        color="yellow",
        label="Observation Set (x, y) - Observer Fault"
    )
    plt.fill_between(
        [starting_observation.lower[0], starting_observation.upper[0]],
        [starting_observation.lower[1], starting_observation.lower[1]],
        [starting_observation.upper[1], starting_observation.upper[1]],
        color="red",
        label="Starting Observation"
    )
    if new_observation_interval:
        plt.fill_between(
            [new_observation_interval.lower[0], new_observation_interval.upper[0]],
            [new_observation_interval.lower[1], new_observation_interval.lower[1]],
            [new_observation_interval.upper[1], new_observation_interval.upper[1]],
            color="cyan",
            label="Interval Containing New Observation"
        )    
    try:
        plt.plot(new_observation[0], new_observation[1], "mo", label="Actual New Observation")
    except:
        pass
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Observation of x and y Positions after 1 Second")
    plt.legend()
    plt.grid()
    plt.show()

def get_noisy_measurement(x, delta, key):
    """
    Returns a random interval that is δ-wide and contains the true output.

    Args:
        x_interval (array): The true output interval (lower and upper bounds).
        delta (float): The noise level (width of the interval).
        key (jax.random.PRNGKey): Random key for JAX's random number generator.

    Returns:
        Interval: A noisy interval containing the true output.
    """


    # Generate random noise for lower and upper bounds
    key, subkey = random.split(key)
    noise_lower = random.uniform(subkey, shape=x.shape, minval=-delta, maxval=0)  # Noise for lower bound
    key, subkey = random.split(key)
    noise_upper = random.uniform(subkey, shape=x.shape, minval=0, maxval=delta)  # Noise for upper bound

    # Compute noisy bounds
    noisy_lower = x + noise_lower
    noisy_upper = x + noise_upper


    # Return the noisy interval
    return irx.Interval(lower=noisy_lower, upper=noisy_upper)

import copy

def visualize_trajectory_given_u_K(x0_interval, u_ol, K, w_interval, p_no_disturbance, p_actuator_fault, dt, observer_offset=jnp.zeros(4), max_iter=100):
    # fault_cases = ["actuator", "observer"]  # Initial fault cases
    x_interval = x0_interval
    x_actual = jnp.zeros(4)
    t = 0
    nominal_car_system = FaultyNonHolonomicCar()
    faulty_car_system = FaultyNonHolonomicCar()
    x_interval_no_fault = copy.deepcopy(x_interval)
    x_interval_observer_fault = copy.deepcopy(x_interval)
    x_interval_actuator_fault = copy.deepcopy(x_interval)
    for i in range(max_iter):

        previous_observation_interval = x_interval
        previous_state = x_actual
        u = u_ol + K @ (x_interval.lower[:2] + x_interval.upper[:2]) / 2.0  # Control input based on the center of the interval
        # Propagate system with optimal control input
        x_interval_no_fault = propagate_interval_euler(irx.natemb(nominal_car_system), x_interval_no_fault, u, w_interval, p_no_disturbance, dt)
        x_interval_actuator_fault = propagate_interval_euler(irx.natemb(faulty_car_system), x_interval_actuator_fault, u, w_interval, p_actuator_fault, dt)
        x_interval_observer_fault = x_interval_no_fault + observer_offset
        def u_map_in_func(t, x):
            return u
        # Define the disturbance and parameter maps
        def w_map_in_func(t, x):
            return jnp.array([0., 0.])
        def p_map_in_func(t, x):
            return jnp.array([0.])
        real_traj = real_car_system.compute_trajectory(0.0, t + dt, previous_state, (u_map_in_func, w_map_in_func, p_map_in_func))
        x_actual = real_traj.ys[-1]
        print(f"x_interval_actuator_fault: {x_interval_actuator_fault[0:2]}")
        print(f"x_interval_observer_fault: {x_interval_observer_fault[0:2]}")
        overlap_size_this_i = overlap_size_lax(x_interval_actuator_fault[0:2], x_interval_observer_fault[0:2])
        # Obtain noisy measurement (δ-wide interval)
        print(f"At time step {i}, overlap size is {overlap_size_this_i}")
        measurement_interval = get_noisy_measurement(x_actual, delta=0.02, key=rng_key)

        t += dt

        print(f"Time: {t}s")
        print(f"Applied control effort: {u}")
        print(f"State interval (no fault): {x_interval_no_fault[0:2]}")
        print(f"State interval (actuator fault): {x_interval_actuator_fault[0:2]}")
        print(f"State interval (observer fault): {x_interval_observer_fault[0:2]}")
        plot_observation_boxes(
            x_interval_no_fault=x_interval_no_fault, 
            x_interval_actuator_fault=x_interval_actuator_fault, 
            x_interval_observation_fault=x_interval_observer_fault,
            starting_observation=previous_observation_interval,
            new_observation_interval=measurement_interval,
            new_observation=x_actual[0:2]
        )

        # Update state interval
        x_interval = measurement_interval



