import immrax as irx
from immrax.utils import null_space
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial
import cvxpy as cp
from matplotlib.patches import Ellipse

@jax.jit
def norm_P (x:jax.Array, P:jax.Array) -> float :
    """Returns :math:`\|x\|_{2,P^{1/2}} = x^T P x`"""
    return jnp.sqrt(x.T @ P @ x)

@register_pytree_node_class
class Ellipsoid :
    """1-level set of V(x) = (x-xc)^T P (x-xc)"""
    P: jax.Array
    xc: jax.Array
    Pinv: jax.Array
    def __init__(self, P:jax.Array, xc:jax.Array|None = None, Pinv:jax.Array|None = None) :
        self.P = P
        self.xc = jnp.asarray(xc) if xc is not None else jnp.zeros(P.shape[0])
        self.Pinv = Pinv if Pinv is not None else jnp.linalg.inv(P)
        # self._X = None
        # self._Y = None
        # self._Z = None
        self.S, self.U = jnp.linalg.eigh(P)
        self.Sinv = 1/self.S

    def V(self, x:jax.Array) -> float :
        return (x - self.xc).T @ self.P @ (x - self.xc)

    def tree_flatten (self) :
        return ((self.P,self.xc,self.Pinv), 'Ellipsoid')
    @classmethod
    def tree_unflatten(cls, aux_data, children) :
        return cls(*children)

    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        n = self.P.shape[0]
        if n == 2 :
            self.plot_ellipse(ax, rescale, **kwargs)
            return
        ind = [k for k in range(n) if k not in [xi,yi]]
        Phat = self.P[ind,:]
        N = null_space(Phat)
        M = N[(xi,yi),:] # Since M is guaranteed 2x2,
        Minv = (1/(M[0,0]*M[1,1] - M[0,1]*M[1,0]))*jnp.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])
        Q = Minv.T@N.T@self.P@N@Minv
        Ellipsoid(Q, self.xc[(xi,yi),]).plot_ellipse(ax, rescale, **kwargs)

    def plot_ellipse (self, ax, rescale=False, **kwargs) :
        n = self.P.shape[0]
        if n != 2 :
            raise ValueError("Can only plot ellipses in 2D, use plot_projection")
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('fill', False)
        width, height = 2*jnp.sqrt(self.Sinv)
        angle = jnp.arctan2(self.U[1, 0], self.U[0, 0]) * 180 / jnp.pi
        ellipse = Ellipse(xy=self.xc, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
        if rescale :
            print(width, height)
            ax.set_xlim(self.xc[0] - 1.5*width, self.xc[0] + 1.5*width)
            ax.set_ylim(self.xc[1] - 1.5*height, self.xc[1] + 1.5*height)
    
    def __repr__(self) :
        return f'Ellipsoid(P={self.P}, xc={self.xc})'
    
    def __str__(self) :
        return f'Ellipsoid(P={self.P}, xc={self.xc})'

def iover (e:Ellipsoid) -> irx.Interval :
    """Interval over-approximation of an Ellipsoid"""
    overpert = jnp.sqrt(jnp.diag(e.Pinv))
    return irx.icentpert(e.xc, overpert)

def eover (ix:irx.Interval, P:jax.Array) -> Ellipsoid :
    """Ellipsoid over-approximation of an Interval"""
    xc, xp = irx.i2centpert(ix)
    corns = irx.get_corners(ix - xc)
    m = jnp.max(jnp.array([norm_P(c, P) for c in corns]))
    return Ellipsoid(P/(m**2), xc)

def reach_P_contraction (rollout, e0:Ellipsoid, N:int, dt:float) :
    """
    For now, rollout is a function that takes x0 (interval) and returns xx, xnom, M.
    """

    ee = [e0]
    for i in range(N) :
        ei = ee[i]
        Pi = ei.P

        ix0 = iover(ee[i])
        xx1, xnom1, M1 = rollout(irx.i2ut(ix0))
        xx2, xnom2, M2 = rollout(xnom1[-1])
