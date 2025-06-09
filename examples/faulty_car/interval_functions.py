import immrax as irx
import jax
import jax.numpy as jnp
import jax.lax as lax

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