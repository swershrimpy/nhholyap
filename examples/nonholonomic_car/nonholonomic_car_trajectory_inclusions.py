import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple
import numpy as np
import sys
import os
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pickle
import pandas as pd
from functools import reduce

# Add the parent directory of 'nhholyap' to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.append("~/output_feedback/nhholyap/examples")

from duality_clf import find_lyapunov_and_gain, find_lyapunov_and_gain_polytope
from nhholyap.examples.nonholonomic_car.nonholonomic_car import NonHolonomicCar, get_polytope_list_from_corners

# First, read in nominal trajectory from CSV file.
filename = "/home/user/output_feedback/nhholyap/nonholonomic_car_trajectory.csv"
data = pd.read_csv(filename)

X_PERTURBATION = np.array([.1, .1, 0.1, .1])  # Perturbation bounds for state
mu = np.array([10, 10])  # Perturbation bounds for control

# Extract states (X, Y, Phi, V) and control inputs (Omega, A)
x_ref = data[["X", "Y", "Phi", "V"]].to_numpy()  # Nx4 array
u_ref = data[["Omega", "A"]].fillna(0).to_numpy()  # Nx2 array, fill NaN with 0 for control inputs

print("x_ref shape:", x_ref.shape)  # Should be Nx4
print("u_ref shape:", u_ref.shape)  # Should be Nx2
print("x_ref[-1]: ", x_ref[-1])

num_knots = x_ref.shape[0]

# Initialize lists to collect matrices and track indices
A_ldi_list = []
B_ldi_list = []
x_bound_list = []  # New list to store state bounds at each knot point

for i in range(num_knots):
    car = NonHolonomicCar()
    x_nominal = x_ref[i, :]
    
    sys_mjacM = irx.mjacM(car.f)
    Ms = sys_mjacM(
        jnp.array([0.0]),  # Bound for t. Any trivial bound works for TI system.
        irx.icentpert(x_nominal, X_PERTURBATION),  # Perturbation on x
        irx.icentpert(jnp.zeros(2), mu),  # Perturbation on u
        irx.icentpert(jnp.zeros(1), jnp.zeros(1)),  # Perturbation on w
        centers=(
            (
                jnp.array([0.]),  # Center for t. Does not matter
                x_nominal,  # Center of x
                jnp.array([0., 0.]),  # Center of u
                jnp.zeros(1),
            ),))
    
    # Extract LDI matrices
    ldi_A = Ms[0][1]  # LDI Set of A matrices
    ldi_B = Ms[0][2]  # LDI Set of B matrices
    ldi_x_bound = irx.icentpert(x_nominal, X_PERTURBATION)  # Extract state bounds

    # Save A, B LDIs, and state bounds
    A_ldi_list.append(ldi_A)
    B_ldi_list.append(ldi_B)
    x_bound_list.append(ldi_x_bound)

# Compute union of LDIs across all knot points
A_ldi_union = irx.Interval(
    lower=reduce(jnp.minimum, [A.lower for A in A_ldi_list]), 
    upper=reduce(jnp.maximum, [A.upper for A in A_ldi_list])
)
B_ldi_union = irx.Interval(
    lower=reduce(jnp.minimum, [B.lower for B in B_ldi_list]), 
    upper=reduce(jnp.maximum, [B.upper for B in B_ldi_list])
)

print(A_ldi_union)

# Get polytopic representation of A and B
A_union_corners = irx.get_sparse_corners(A_ldi_union)
B_union_corners = irx.get_sparse_corners(B_ldi_union)
A_list, B_list = get_polytope_list_from_corners(A_corners=A_union_corners, B_corners=B_union_corners)

# Save all computed data
with open("polytope_data.pkl", "wb") as f:
    pickle.dump({
        "A_list": A_list, 
        "B_list": B_list, 
        "A_ldi_list": A_ldi_list, 
        "B_ldi_list": B_ldi_list,
        "x_bound_list": x_bound_list  # Save state bounds
    }, f)

print(A_list)
