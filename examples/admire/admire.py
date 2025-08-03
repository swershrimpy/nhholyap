import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple

class AdmireNineDoFLinAct(irx.system.System):
    """
    Immrax system for the Admire example.
    9 Degrees of Freedom, Lg is linearized.
    \dot{x} = f(x) + g(x) * u
    where:
    g(x) = B as described in Bouvier and Ornik. 
    """
    def __init__(self) -> None:
        # Tells immrax that the system is continuous
        self.evolution = 'continuous'
        # Tells immrax the number of states
        self.xlen = 9  #vt, alpha, beta, pb, qb, rb, psi, theta, phi, xv, yv, zv, ub, vb, wb
        self.ulen = 10  #cc, cl, cm, cn, fx, fy, my
        # self.wlen = 1  #disturbance
        # self.vlen = 3  #noise in output
        # self.plen = 1
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
        self.dIzz = 0.0
        self.dIxx = 0.0
        self.dIyy = 0.0
        self.dIxz = 0.0
        # Calculate coefficients
        self.Gama = self.Ix*(1+self.dIxx)*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz)						
        self.C1	 = ((self.Iy*(1+self.dIyy)-self.Iz*(1+self.dIzz))*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/self.Gama        
        self.C2	 = ((self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy)+self.Iz*(1+self.dIzz))*self.Ixz*(1+self.dIxz))/self.Gama
        self.C3   = self.Iz*(1+self.dIzz)/self.Gama
        self.C4	 = self.Ixz*(1+self.dIxz)/self.Gama
        self.C5   = (self.Iz*(1+self.dIzz)-self.Ix*(1+self.dIxx))/(self.Iy*(1+self.dIyy))
        self.C6	 = self.Ixz*(1+self.dIxz)/(self.Iy*(1+self.dIyy))
        self.C7	 = 1/(self.Iy*(1+self.dIyy))
        self.C8	 = (self.Ix*(1+self.dIxx)*(self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy))+self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/self.Gama
        self.C9	 = (self.Ix*(1+self.dIxx))/self.Gama
        # Define the input matrix B
        #B_bar = [-1.42042676365169,-1.42042676365169,-0.927171070762340,-1.42757288184381,-1.42757288184381,-0.927171070762340,-0.307532766010299,0.448846619230885,-0.703180119671139,6.01815430830781,-15.3048650281407,-50.9506691491745;-0.000549876720983788,-0.000549876720983788,-0.0566965119412683,-0.0909778752323707,-0.0909778752323707,-0.0566965119412683,0.000357863410678141,0.00520822790871878,0.00179181907856105,-0.00699627606405290,0.0178803821307243,-3.04799422687635;-0.00633411409606306,0.00633411409606306,0.00353654046969741,0.0148728219315439,-0.0148728219315439,-0.00353654046969741,0.0441553601137058,-1.37664366754032e-51,4.29199939146374e-101,-1.33812831967177e-150,3.08653189849630,1.88358943903008e-16;0.849549651283458,-0.849549651283458,-5.20922170298741,-4.48930840943736,4.48930840943736,5.20922170298741,3.03540858873301,-1.40744369270775e-50,4.38802546734135e-100,-1.36806663043070e-149,-18.2188628376002,-1.11182578896268e-15;1.54036208811258,1.54036208811258,-1.25649744402598,-2.01079851628958,-2.01079851628958,-1.25649744402598,0.00519059193586529,-0.137932956532161,-0.143274324668438,-0.101401906195261,0.259657357025434,-190.001762258710;-0.435552469139453,0.435552469139453,-0.230910554336885,-0.492831044428952,0.492831044428952,0.230910554336885,-1.83801900441132,9.55556432454017e-51,-2.97916426981405e-100,9.28822144364958e-150,-153.034955897234,-9.33912352796015e-15;0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0];
        # The matrix data is first defined as a standard Python list of lists.
        
        ## This is from J-B Bouvier's code.
        # data_list = [
        #     [-1.42042676365169, -1.42042676365169, -0.927171070762340, -1.42757288184381, -1.42757288184381, -0.927171070762340, -0.307532766010299, 0.448846619230885, -0.703180119671139, 6.01815430830781, -15.3048650281407, -50.9506691491745],
        #     [-0.000549876720983788, -0.000549876720983788, -0.0566965119412683, -0.0909778752323707, -0.0909778752323707, -0.0566965119412683, 0.000357863410678141, 0.00520822790871878, 0.00179181907856105, -0.00699627606405290, 0.0178803821307243, -3.04799422687635],
        #     [-0.00633411409606306, 0.00633411409606306, 0.00353654046969741, 0.0148728219315439, -0.0148728219315439, -0.00353654046969741, 0.0441553601137058, -1.37664366754032e-51, 4.29199939146374e-101, -1.33812831967177e-150, 3.08653189849630, 1.88358943903008e-16],
        #     [0.849549651283458, -0.849549651283458, -5.20922170298741, -4.48930840943736, 4.48930840943736, 5.20922170298741, 3.03540858873301, -1.40744369270775e-50, 4.38802546734135e-100, -1.36806663043070e-149, -18.2188628376002, -1.11182578896268e-15],
        #     [1.54036208811258, 1.54036208811258, -1.25649744402598, -2.01079851628958, -2.01079851628958, -1.25649744402598, 0.00519059193586529, -0.137932956532161, -0.143274324668438, -0.101401906195261, 0.259657357025434, -190.001762258710],
        #     [-0.435552469139453, 0.435552469139453, -0.230910554336885, -0.492831044428952, 0.492831044428952, 0.230910554336885, -1.83801900441132, 9.55556432454017e-51, -2.97916426981405e-100, 9.28822144364958e-150, -153.034955897234, -9.33912352796015e-15],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # ]

        ## This is from J-B Bouvier's paper.
        data_list = [
            [-.62, -.62, -.4, -.62, -.62, -.4, -.16, .08, -.53, -1.78],
            [0., 0., -.02, -.04, -.04, -.02, 0., 0., 0., -.11],
            [0., 0., 0., .01, -.01, 0., .02, 0., .11, .0],
            [.37, -.37, -2.27, -1.96, 1.96, 2.27, 1.59, 0., -.64, 0.],
            [.67, .67, -.55, -.88, -.88, -.55, 0., -.02, .01, -6.63],
            [-.19, .19, -.1, -.22, .22, .1, -.96, .0, -5.34, 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]
        # Convert the list of lists into a JAX NumPy array.
        self.B_bar = jnp.array(data_list)
                    


    def f(self, t: float, x: jax.Array, u: jax.Array, p: jax.Array) -> jax.Array:
        # assert x.shape == (self.xlen,), f"Expected x to be of shape ({self.xlen},), got {x.shape}"
        # assert u.shape == (self.ulen,), f"Expected u to be of shape ({self.ulen},), got {u.shape}"
        # assert w.shape == (self.wlen,), f"Expected w to be of shape ({self.wlen},), got {w.shape}"
        # assert p.shape == (self.plen,), f"Expected p to be of shape ({self.plen},), got {p.shape}"
        # Unpack the state vector following convention in C++ code.
        Vt_st, alpha_st, beta_st, pb_st, qb_st, rb_st, psi_st, theta_st, phi_st = x[:9]
        # Unpack inputs
        # drc_in, dlc_in, droe_in, drie_in, dlie_in, dloe_in, dr_in. dle_in, ldg_in, tss_in, dty_in, dtz_in, u_dist, v_dist, w_dist, p_dist = u[:15]
        # TODO: Unpack disturbance
        # Calculate Derivatives 
        ubody = Vt_st * jnp.cos(alpha_st) * jnp.cos(beta_st)
        vbody = Vt_st * jnp.sin(beta_st)
        wbody = Vt_st * jnp.sin(alpha_st) * jnp.cos(beta_st)
        uderiv = rb_st * vbody - qb_st * wbody - self.gravity * jnp.sin(theta_st) 
        vderiv = - rb_st * ubody + pb_st * wbody + self.gravity * jnp.sin(phi_st) * jnp.cos(theta_st)
        wderiv = qb_st * ubody - pb_st * vbody + self.gravity * jnp.cos(phi_st) * jnp.cos(theta_st)
        Vt_der = (ubody*uderiv+vbody*vderiv+wbody*wderiv)/Vt_st
        alpha_der = (ubody*wderiv-wbody*uderiv)/(ubody*ubody+wbody*wbody)
        beta_der  = (vderiv*Vt_st-vbody*Vt_der)/(Vt_st*Vt_st*jnp.cos(beta_st))
        pb_der    = (self.C1*rb_st+self.C2*pb_st)*qb_st
        qb_der    =  self.C5*pb_st*rb_st-self.C6*(pb_st*pb_st-rb_st*rb_st)
        rb_der    = (self.C8*pb_st-self.C2*rb_st)*qb_st

        psi_der   = (qb_st*jnp.sin(phi_st)+rb_st*jnp.cos(phi_st))/jnp.cos(theta_st)
        theta_der =  qb_st*jnp.cos(phi_st)-rb_st*jnp.sin(phi_st)
        phi_der   =  pb_st+jnp.tan(theta_st)*(qb_st*jnp.sin(phi_st)+rb_st*jnp.cos(phi_st))

        internal_dynamics = jnp.array([
            Vt_der,
            alpha_der,
            beta_der,
            pb_der,
            qb_der,
            rb_der,
            psi_der,
            theta_der,
            phi_der
        ])

        actuator_dynamics = self.B_bar @ (jnp.diag(p) @ u[:self.ulen])  # Assuming B is defined in the system

        return internal_dynamics + actuator_dynamics 
    
    def g(self, t: float, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Definition of Output.
        """
        Vt_st, alpha_st, beta_st, pb_st, qb_st, rb_st, psi_st, theta_st, phi_st = x[:9]
        
        ubody = Vt_st * jnp.cos(alpha_st) * jnp.cos(beta_st)
        vbody = Vt_st * jnp.sin(beta_st)
        wbody = Vt_st * jnp.sin(alpha_st) * jnp.cos(beta_st)
        uv = jnp.cos(phi_st) * jnp.cos(psi_st) * ubody + \
            (jnp.sin(phi_st) * jnp.sin(theta_st) * jnp.cos(psi_st) - jnp.cos(phi_st) * jnp.sin(psi_st)) * vbody + \
            (jnp.cos(phi_st) * jnp.sin(theta_st) * jnp.cos(psi_st) + jnp.sin(phi_st) * jnp.sin(psi_st)) * wbody
        vv = jnp.cos(phi_st) * jnp.sin(psi_st) * ubody + \
            (jnp.sin(phi_st) * jnp.sin(theta_st) * jnp.sin(psi_st) + jnp.cos(phi_st) * jnp.cos(psi_st)) * vbody + \
            (jnp.cos(phi_st) * jnp.sin(theta_st) * jnp.sin(psi_st) - jnp.sin(phi_st) * jnp.cos(psi_st)) * wbody
        wv = -jnp.sin(theta_st) * ubody + jnp.sin(phi_st) * jnp.cos(theta_st) * vbody + jnp.cos(phi_st) * jnp.cos(theta_st) * wbody
        return jnp.concatenate(x, jnp.array([uv, vv, wv, pb_st, qb_st, rb_st])) + v
        