import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import numpy as np
import sympy as sp
from sympy import Matrix, Transpose, Poly
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

def dcn(x):
    return x.detach().cpu().numpy()

def gen_z_mon(n, deg):
    """
    Generate monomials for the z vector
    :param n: number of states
    :param deg: degree of the polynomial
    :return: z_u: list of monomials
    """
    x = list(sp.symbols('x[0:%d]' % n))
    x_rev = x.copy()
    x_rev.reverse()
    z_u = sorted(itermonomials(x, deg), key=monomial_key('grlex', x_rev))
    return str(z_u[1:]).replace("x[", "x[:,"), z_u[1:]

def z_mon_eval(x, z_str):
    return torch.vstack(eval(z_str))
