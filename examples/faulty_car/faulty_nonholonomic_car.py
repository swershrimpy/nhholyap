import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple
import numpy as np
import sys
import os

class FaultyNonHolonomicCar(irx.system.OpenLoopSystem):
    p:float
    """
    System Model of a "Faulty" Nonholonomic car.
    Dynamics:
    ---
    \dot{
        p_x,     v * cos(phi)     0   
        p_y,  =  v * sin(phi)   + 0
        phi,     0                omega
        v        0                a
        }
    ---
    """

    def __init__(self) -> None:
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 4  #px, py, phi, v
        self.ulen = 2  #omega, a
        self.wlen = 1  #disturbance
        self.vlen = 3  #noise in output
        # self.p = p # Amount of steering control you have. IDEALLY (?) Between 0 and 1. NOTE: treated as an input instead.
        self.plen = 1

    def f(self, t: float, x: jax.Array, u: jax.Array, w: jax.Array, p: jax.Array) -> jax.Array:
        assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"
        assert u.shape == (self.ulen,), f"Expected u to be of shape ({self.ulen},), got {u.shape}"
        assert w.shape == (self.wlen,), f"Expected w to be of shape ({self.wlen},), got {w.shape}"
        assert p.shape == (self.plen,), f"Expected p to be of shape ({self.plen},), got {p.shape}"
        return jnp.array([
            x[3] * jnp.cos(x[2]),
            x[3] * jnp.sin(x[2]),
            p[0] * u[0] + w[0],
            u[1] + w[1],
        ])
    
    def h(self, t: float, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Definition of Observer.
        y = Cx + v
        Only px, py and steering angle can be observed.
        """
        return jnp.array([
            x[0] + v[0],
            x[1] + v[1], 
            x[2] + v[2],
        ])