import immrax as irx
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Union


def overlap_size_lax(interval1, interval2):
    """Compute the volume of the intersection of two intervals (JAX-compatible)."""
    intersection_lower = jnp.maximum(interval1.lower, interval2.lower)
    intersection_upper = jnp.minimum(interval1.upper, interval2.upper)

    def has_overlap(operands):
        lower, upper = operands
        return jnp.prod(upper - lower)

    def no_overlap(operands):
        _ = operands
        return jnp.array(0.0)

    has_no_overlap = jnp.any(intersection_upper < intersection_lower)

    return lax.cond(
        jnp.logical_not(has_no_overlap),
        has_overlap,
        no_overlap,
        (intersection_lower, intersection_upper),
    )


def propagate_interval_euler(
    embedding_system,
    x0_interval,
    u_interval,
    w_interval,
    p_interval,
    t_end,
):
    """Forward Euler interval propagation for one time step."""
    xt_ut = (
        embedding_system.f(0.0, irx.i2ut(x0_interval), u_interval, w_interval, p_interval)
        * t_end
        + irx.i2ut(x0_interval)
    )
    return irx.ut2i(xt_ut)


def propagate_interval(
    embedding_system,
    x0_interval,
    u_interval,
    w_interval,
    p_interval,
    t_end,
):
    """Full ODE interval propagation using immrax compute_trajectory."""
    def u_map(t, x):
        return u_interval

    def w_map(t, x):
        return w_interval

    def p_map(t, x):
        return p_interval

    x_traj = embedding_system.compute_trajectory(
        0.0, t_end, irx.i2ut(x0_interval), (u_map, w_map, p_map)
    )
    x_emb = x_traj.ys[-1]
    n = x0_interval.lower.shape[0]
    return irx.Interval(lower=x_emb[:n], upper=x_emb[n:])
