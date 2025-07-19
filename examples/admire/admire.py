import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple

def Admire(irx.system.System):
    """
    Immrax system for the Admire example.
    """
    def __init__(self) -> None:
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 15  #vt, alpha, beta, pb, qb, rb, psi, theta, phi, xv, yv, zv, ub, vb, wb
        self.ulen = 2  #cc, cl, cm, cn, fx, fy, my
        self.wlen = 1  #disturbance
        self.vlen = 3  #noise in output
        self.plen = 1
        # Define model parameters
        self.mass = 9100. 
        self.Ix = 21000.
        self.Iy = 81000.
        self.Iz = 101000.
        self.Ixz = 2500.
        self.Sref = 45.00 # wing area
        self.bref = 10.00 # wingspan
        self.cref = 5.20  # mean aerodynamic chord
        self.gravity = 9.81  # gravitational acceleration
        # Define deviations
        self.dmass = 0.
        self.dIx = 0.0
        self.dIy = 0.0
        self.dIz = 0.0
        self.dIxz = 0.0
        # Calculate coefficients
        Gama = self.Ix*(1+self.dIxx)*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz)						
        C1	 = ((self.Iy*(1+self.dIyy)-self.Iz*(1+self.dIzz))*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/Gama        
        C2	 = ((self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy)+self.Iz*(1+self.dIzz))*self.Ixz*(1+self.dIxz))/Gama
        C3   = self.Iz*(1+self.dIzz)/Gama
        C4	 = self.Ixz*(1+self.dIxz)/Gama
        C5   = (self.Iz*(1+self.dIzz)-self.Ix*(1+self.dIxx))/(self.Iy*(1+self.dIyy))
        C6	 = self.Ixz*(1+self.dIxz)/(self.Iy*(1+self.dIyy))
        C7	 = 1/(self.Iy*(1+self.dIyy))
        C8	 = (self.Ix*(1+self.dIxx)*(self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy))+self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/Gama
        C9	 = (self.Ix*(1+self.dIxx))/Gama


    def f(self, t: float, x: jax.Array, u: jax.Array, w: jax.Array, p: jax.Array) -> jax.Array:
        # assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"
        # assert u.shape == (self.ulen,), f"Expected u to be of shape ({self.ulen},), got {u.shape}"
        # assert w.shape == (self.wlen,), f"Expected w to be of shape ({self.wlen},), got {w.shape}"
        # assert p.shape == (self.plen,), f"Expected p to be of shape ({self.plen},), got {p.shape}"
        # Unpack the state vector following convention in C++ code.
        Vt_st, alpha_st, beta_st, pb_st, qb_st, rb_st, psi_st, theta_st, phi_st, x_st, y_st, z_st = x[:12]
        # Unpack inputs
        drc_in, dlc_in, droe_in, drie_in, dlie_in, dloe_in, dr_in. dle_in, ldg_in, tss_in, dty_in, dtz_in, u_dist, v_dist, w_dist, p_dist = u[:15]
        # TODO: Unpack disturbance
        # Calculate Derivatives 
        """
        uderiv =  rb_st*vbody-qb_st*wbody-grav*sin(theta_st)+(Fx+Tx)/(mass*(1+dmass));
        vderiv = -rb_st*ubody+pb_st*wbody+grav*sin(phi_st)*cos(theta_st)+(Fy+Ty)/(mass*(1+dmass));
        wderiv =  qb_st*ubody-pb_st*vbody+grav*cos(phi_st)*cos(theta_st)+(Fz+Tz)/(mass*(1+dmass));
        """
        vtd = (ub * ubd + vb * vbd +)



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