import jax.numpy as jnp
from visualization_functions import visualize_trajectory_given_u_K
import immrax as irx
RGX = jnp.array([-0.13220807, -0.80112254, -0.00476904, -0.00217136, -0.03877236, -0.01650559])
u_ol = RGX[0:2]
K = jnp.array([
    [RGX[2], RGX[3]],
    [RGX[4], RGX[5]]
])
# Now you can use u_ol and K in your control system
# Propagate the system with the optimal control input

w_fixed = jnp.array([0., 0.])  # Fixed disturbance. Assume steering angle is fixed at 45 deg.
w_interval = irx.icentpert(w_fixed, jnp.zeros_like(w_fixed))
p_interval = irx.icentpert(jnp.array([.25]), jnp.array([.25]))
p_no_disturbance = irx.icentpert(jnp.array([1]), jnp.array([.0]))
# visualize_trajectory_given_u_K(
#     x0_interval=irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.02, 0.02, 0.02, 0.02])),
#     u_ol=u_ol,
#     K=K,
#     dt=1.0,
#     w_interval=w_interval,
#     p_no_disturbance=p_no_disturbance,
#     p_actuator_fault=p_interval,
#     max_iter=10,
# )

visualize_trajectory_given_u_K(
    x0_interval=irx.interval(jnp.array([0., 0., 0., 0.]), jnp.array([0.02, 0.02, 0.02, 0.02])),
    u_ol=jnp.array([-0.38392818, -0.92336296]),
    K=jnp.zeros_like(K),
    dt=1.0,
    w_interval=w_interval,
    p_no_disturbance=p_no_disturbance,
    p_actuator_fault=p_interval,
    observer_offset=jnp.ones(4) * 0.2,  # Offset for the observer
    max_iter=3,
)