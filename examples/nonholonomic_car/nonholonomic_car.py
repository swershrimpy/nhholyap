import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple
import numpy as np
import sys
import os
# Add the parent directory of 'nhholyap' to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "nhholyap"))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from duality_clf import find_lyapunov_and_gain, find_lyapunov_and_gain_polytope
import copy

print(jax.devices())

def get_polytope_list_from_corners(A_corners, B_corners):
    A_list = []
    B_list = []
    for A_corner in A_corners:
        for B_corner in B_corners:
            A_list.append(copy.deepcopy(A_corner))
            B_list.append(copy.deepcopy(B_corner))
    return A_list, B_list


class NonHolonomicCar(irx.system.OpenLoopSystem):
    m:float
    l:float
    b:float

    def __init__(self) -> None:
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 4  #px, py, phi, v
        self.ulen = 2  #omega, a
        self.wlen = 1  #disturbance

    def f(self, t: float, x: jax.Array, u: jax.Array, w: jax.Array) -> jax.Array:
        assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"
        assert u.shape == (self.ulen,), f"Expected u to be of shape ({self.ulen},), got {u.shape}"
        assert w.shape == (self.wlen,), f"Expected w to be of shape ({self.wlen},), got {w.shape}"
        return jnp.array([
            x[3] * jnp.cos(x[2]),
            x[3] * jnp.sin(x[2]),
            u[0] + w[0],
            u[1] + w[1],
        ])