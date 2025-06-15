import immrax as irx
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Union

def overlap_size(interval1, interval2):
    # Compute the intersection lower and upper bounds
    intersection_lower = jnp.maximum(interval1.lower, interval2.lower)
    intersection_upper = jnp.minimum(interval1.upper, interval2.upper)
    print(intersection_lower)
    print(intersection_upper)
    # Calculate the size of the intersection
    if jnp.any(intersection_upper < intersection_lower):
        return 0
    else:
        intersection_size = jnp.prod(intersection_upper - intersection_lower)
        return intersection_size
    

# Your new, JAX-compatible function
def overlap_size_lax(interval1, interval2):
    # Calculate intersection bounds (this part is fine)
    intersection_lower = jnp.maximum(interval1.lower, interval2.lower)
    intersection_upper = jnp.minimum(interval1.upper, interval2.upper)

    # Define the two functions for the conditional branches.
    # They must take the same arguments (the 'operands').
    def has_overlap(operands):
        # Operands are the values needed for the calculation
        lower, upper = operands
        return jnp.prod(upper - lower)

    def no_overlap(operands):
        # This branch doesn't need the operands, but must accept them.
        _ = operands
        return jnp.array(0.0)

    # The condition that determines which function to run
    # This evaluates to a JAX boolean tracer
    has_no_overlap = jnp.any(intersection_upper < intersection_lower)

    # Use lax.cond to choose which branch to execute.
    # Note that we are using `jnp.logical_not` to flip the condition
    # to match the `has_overlap` function.
    return lax.cond(
        jnp.logical_not(has_no_overlap), # Condition: if there IS overlap
        has_overlap,                    # Function to run if True
        no_overlap,                     # Function to run if False
        (intersection_lower, intersection_upper) # Operands to pass to the chosen function
    )

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

def propagate_with_feedback(x_interval, u_ol, K, dt, faulty_system, w_if, p_ivl):
    """Propagate the state interval under feedback control u = Kx."""
    # Assuming x_interval is a 4D state (lower and upper bounds for each state variable)
    # and K is a 2x4 matrix (since u is 2D and x is 4D).
    # For simplicity, we'll approximate the propagation of the interval under u = Kx.
    # This is a placeholder; you may need a more rigorous interval propagation method.
    x_center = (x_interval.lower + x_interval.upper) / 2
    u = u_ol +jnp.array([0., 1.]) + K @ x_center[:2]  # Feedback control
    # w_if = irx.icentpert(jnp.array([0, 0]), jnp.zeros(2))  # Assuming no steady-state disturbance on input for simplicity
    x_interval_propagated = propagate_interval_euler(
        irx.natemb(faulty_system), x_interval, irx.icentpert(u, jnp.zeros_like(u)), w_if, p_ivl, dt
    )
    return x_interval_propagated

def observation_interval_size(x_interval):
    """Compute the size of the observation interval (sum of interval widths)."""
    # x_interval is assumed to be a 2D array where each row is [lower, upper] for a state variable.
    widths = x_interval.upper - x_interval.lower
    return jnp.prod(widths)