import immrax as irx
import jax
import jax.numpy as jnp

class VanDerPolOscillator1D(irx.system.System):
    mu: float

    def __init__(self, mu:float=1) -> None:
        super().__init__()
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 2
        self.mu = mu
        self.fi = [
            lambda t, x: x[1],
            lambda t, x: self.mu * (1 - x[0] ** 2) * x[1] - x[0],
        ]

    def f(self, t: float, x: jax.Array) -> jax.Array:
        # TODO: define evolution of controller variable z.
        assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"

        return jnp.array([
            x[1],
            self.mu * (1 - x[0] ** 2) * x[1] - x[0] 
        ])
