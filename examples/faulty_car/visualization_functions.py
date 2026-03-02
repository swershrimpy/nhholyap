import immrax as irx
import jax
import jax.numpy as jnp
from jax import random
from faulty_nonholonomic_car import FaultyNonHolonomicCar
from typing import Union
from immutabledict import immutabledict
import matplotlib.pyplot as plt
from interval_functions import overlap_size, overlap_size_lax
import copy, pickle

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

def plot_observation_boxes(
    x_interval_actuator_fault, 
    x_interval_no_fault, 
    x_interval_observation_fault, 
    starting_observation, 
    x_actual_hist=None, 
    new_observation_interval=None, 
    new_observation=None, 
    t=1,
    filename="traj_plot.pdf"  # New parameter to specify the output filename
):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- PLOTTING LOGIC (uses ax. instead of plt.) ---
    ax.fill_between(
        [x_interval_actuator_fault[0].lower, x_interval_actuator_fault[0].upper],
        [x_interval_actuator_fault[1].lower, x_interval_actuator_fault[1].lower],
        [x_interval_actuator_fault[1].upper, x_interval_actuator_fault[1].upper],
        alpha=0.5,
        color="blue",
        label="Observation Set (x, y) - Actuator Fault"
    )
    # ... (repeat for your other fill_between calls, using ax. instead of plt.)
    ax.fill_between(
        [x_interval_observation_fault[0].lower, x_interval_observation_fault[0].upper],
        [x_interval_observation_fault[1].lower, x_interval_observation_fault[1].lower],
        [x_interval_observation_fault[1].upper, x_interval_observation_fault[1].upper],
        alpha=0.5,
        color="yellow",
        label="Observation Set (x, y) - Observer Fault"
    )
    ax.fill_between(
        [starting_observation.lower[0], starting_observation.upper[0]],
        [starting_observation.lower[1], starting_observation.lower[1]],
        [starting_observation.upper[1], starting_observation.upper[1]],
        color="red",
        label="Starting Observation"
    )

    if x_actual_hist:
        x_positions = [state[0] for state in x_actual_hist]
        y_positions = [state[1] for state in x_actual_hist]
        ax.plot(x_positions, y_positions, 
                 color='green', 
                #  marker='o', 
                 linestyle='-', 
                 markersize=4, 
                 label="Actual Trajectory")
    
    # --- FONT SIZE MODIFICATIONS ---
    LABEL_FONT_SIZE = 14
    TITLE_FONT_SIZE = 16
    TICK_FONT_SIZE = 12

    ax.set_xlabel("X Position [m]", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Y Position [m]", fontsize=LABEL_FONT_SIZE)
    # ax.set_title(f"Observation of x and y Positions After {t:.2f} seconds", fontsize=TITLE_FONT_SIZE)
    
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    
    # --- LEGEND AND GRID ---
    ax.legend(loc='upper right', fontsize=TICK_FONT_SIZE) # You can also control legend font size
    ax.grid(True)
    
    # --- SAVE OR SHOW THE PLOT ---
    if filename:
        # Save the figure to a file
        # bbox_inches='tight' trims the whitespace around the figure
        # pad_inches controls the padding after trimming
        plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.)
        print(f"Plot saved to {filename}")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Show the plot interactively
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
    print(f"Noise lower: {noise_lower}")
    key, subkey = random.split(key)
    noise_upper = random.uniform(subkey, shape=x.shape, minval=0, maxval=delta)  # Noise for upper bound

    # Compute noisy bounds
    noisy_lower = x + noise_lower
    noisy_upper = x + noise_upper


    # Return the noisy interval
    return irx.Interval(lower=noisy_lower, upper=noisy_upper), key


def visualize_trajectory_given_u_K(
    x0_interval, 
    u_ol, 
    K, 
    w_interval, 
    p_no_disturbance, 
    p_actuator_fault, 
    dt, 
    observer_offset=jnp.zeros(4), 
    max_iter=100,
    output_filename="trajectory_history.pkl"
):
    # Initialize system and states
    x_interval = x0_interval
    x_actual = jnp.array([.1, .1, .0, .0])
    t = 0
    nominal_car_system = FaultyNonHolonomicCar()
    faulty_car_system = FaultyNonHolonomicCar()
    
    x_interval_no_fault = copy.deepcopy(x_interval)
    x_interval_observer_fault = copy.deepcopy(x_interval)
    x_interval_actuator_fault = copy.deepcopy(x_interval)
    
    # Initialize a history dictionary to store JAX/NumPy arrays during the simulation
    # This is efficient for the computation loop.
    history = {
        "no_fault": {"lower": [x_interval_no_fault.lower], "upper": [x_interval_no_fault.upper]},
        "actuator_fault": {"lower": [x_interval_actuator_fault.lower], "upper": [x_interval_actuator_fault.upper]},
        "observer_fault": {"lower": [x_interval_observer_fault.lower], "upper": [x_interval_observer_fault.upper]},
        "measurements": {"lower": [x0_interval.lower], "upper": [x0_interval.upper]},
        "actual_trajectory": [x_actual],
        "controls": []
    }
    
    for_key = rng_key
    
    # --- SIMULATION LOOP (No changes here) ---
    for i in range(max_iter):
        previous_state = x_actual
        u = u_ol + K @ (x_interval.lower[:2] + x_interval.upper[:2]) / 2.0
        
        x_interval_no_fault = propagate_interval_euler(irx.natemb(nominal_car_system), x_interval_no_fault, u, w_interval, p_no_disturbance, dt)
        x_interval_actuator_fault = propagate_interval_euler(irx.natemb(faulty_car_system), x_interval_actuator_fault, u, w_interval, p_actuator_fault, dt)
        x_interval_observer_fault = x_interval_no_fault + observer_offset
        
        real_traj = real_car_system.compute_trajectory(0.0, dt, previous_state, (lambda t, x: u, lambda t, x: jnp.zeros(2), lambda t, x: jnp.zeros(1)))
        x_actual = real_traj.ys[-1]
        
        measurement_interval, for_key = get_noisy_measurement(x_actual, delta=0.02, key=for_key)
        
        # Append JAX/NumPy arrays to the history
        history["no_fault"]["lower"].append(x_interval_no_fault.lower)
        history["no_fault"]["upper"].append(x_interval_no_fault.upper)
        history["actuator_fault"]["lower"].append(x_interval_actuator_fault.lower)
        history["actuator_fault"]["upper"].append(x_interval_actuator_fault.upper)
        history["observer_fault"]["lower"].append(x_interval_observer_fault.lower)
        history["observer_fault"]["upper"].append(x_interval_observer_fault.upper)
        history["measurements"]["lower"].append(measurement_interval.lower)
        history["measurements"]["upper"].append(measurement_interval.upper)
        history["actual_trajectory"].append(x_actual)
        history["controls"].append(u)
        
        print(f"Time: {t}s, Applied control: {u}")
        x_interval = measurement_interval
        t += dt
    # --- END OF SIMULATION LOOP ---

    # --- MODIFICATION START ---
    # 1. Convert all JAX/NumPy arrays in the history to native Python lists before saving.
    # This ensures the output file is portable and has no library dependencies.
    print("\nConverting history to native Python types for saving...")
    native_history = {
        "actual_trajectory": [arr.tolist() for arr in history["actual_trajectory"]],
        "controls": [arr.tolist() for arr in history["controls"]]
    }
    for fault_type in ["no_fault", "actuator_fault", "observer_fault", "measurements"]:
        native_history[fault_type] = {
            "lower": [arr.tolist() for arr in history[fault_type]["lower"]],
            "upper": [arr.tolist() for arr in history[fault_type]["upper"]]
        }
    # --- MODIFICATION END ---
        
    # 2. Save the new 'native_history' dictionary to the pickle file.
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(native_history, f)
        print(f"Successfully saved trajectory history to {output_filename}")
    except Exception as e:
        print(f"\nError saving data to {output_filename}: {e}")

    # Plotting call remains the same, using the final intervals from the simulation
    plot_observation_boxes(
        x_interval_no_fault=x_interval_no_fault, 
        x_interval_actuator_fault=x_interval_actuator_fault, 
        x_interval_observation_fault=x_interval_observer_fault,
        starting_observation=x0_interval,
        x_actual_hist=history["actual_trajectory"], # Use original history with arrays for plotting
        t=t
    )

    return history # Return the original history with arrays for potential further computation