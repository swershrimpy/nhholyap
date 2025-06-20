import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple
import numpy as np
import sys
import os

class TwoTanks(irx.system.System):
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

    def __init__(self, hmax=0.6, a=0.0154) -> None:
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 2  
        self.wlen = 2  #q_pbar , c2
        self.clen = 2 #c_L, c12
        self.hmax = hmax #maximum height of the tank
        self.a = a # #cross-sectional area of the tank
        # self.wlen = 3  #disturbance

    def f(self, t: float, x: jax.Array, w: jax.Array, c: jax.Array) -> jax.Array:
        # assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"
        # assert u.shape == (self.ulen,), f"Expected u to be of shape ({self.ulen},), got {u.shape}"
        # assert w.shape == (self.wlen,), f"Expected w to be of shape ({self.wlen},), got {w.shape}"
        # assert p.shape == (self.plen,), f"Expected p to be of shape ({self.plen},), got {p.shape}"
        qpk = w[0] * (1. - jnp.sqrt(x[0] / self.hmax))
        qlk = c[0] * jnp.sqrt(x[0])
        q12k = c[1] * jnp.sqrt(x[0] - x[1])
        q2k = w[1] * jnp.sqrt(x[1])
        return jnp.array([
            qpk - qlk - q12k,
            q12k - q2k,
        ]) / self.a
    
