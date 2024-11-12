import scipy
import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp
import matplotlib as plt

def calculate_lyapunov_P_matrix_abdelraouf(c, x0, A_list):
    """
    Calculates P matrix for nonhomogeneous HO Lyapunov function according to Hassan Abdelraouf's paper.
    Works for arbitrary dimension - SLOW.
    Inputs:
        c: order of Lyapunov function.
        x0: equilibrium point.
        A_list: list of A matrices in set M.

    Returns:
        P_value: P matrix for nonhomogeneous HO Lyapunov function.
    """
    state_length = A_list[0].shape[0]
    final_P_tilda_dim = int(state_length * (state_length ** c - 1) / (state_length - 1))

    # Construct all B matrices
    Ac_list = [[np.copy(A_list[i]) for i in range(len(A_list))]]
    for i in range(1, c):
        Ac_list.append([np.kron(np.eye(state_length), Ac_list[i-1][_]) + np.kron(A_list[_], np.eye(state_length**i))
                   for _ in range(len(Ac_list[i - 1]))])
    Ac_tilda_list = []
    for j in range(len(Ac_list[0])):
        Ac_tilda_j = []
        for i in range(len(Ac_list)):
            Ac_tilda_j.append(Ac_list[i][j])
        new_Ac_tilda = block_diag(*Ac_tilda_j)
        Ac_tilda_list.append(new_Ac_tilda)
    P = cp.Variable((final_P_tilda_dim, final_P_tilda_dim), symmetric=True)
    constraints = [
        Ac.T @ P + P @ Ac << 0 for Ac in Ac_tilda_list]
    constraints.append(
        P >> np.eye(final_P_tilda_dim)
    )
    xi_list = [np.copy(x0)]
    for i in range(1, c):
        xi_list.append(np.kron(x0, xi_list[-1]))
    xc_0 = np.concatenate(xi_list)
    objective = cp.Minimize(xc_0.T @ P @ xc_0)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if P.value is None:
        raise ValueError("No solution found")

    P_value = P.value   
    return P_value


# TODO: Do a special case. Compute W Matrix for 2D systems. 


def plot_reachable_set_2d(P_value, c, x0, actual_reachable_set_states=None):    
    """
    Plots 2D reachable set of the system from P matrix using Abdelraouf's formulation.
    No dimensionality reduction here. TODO: For 2D systems, dimensionality reduction 
    with W matrix is possible. Implement this.
    Inputs:
        P_value: P matrix of Lyapunov function.
        c: order of Lyapunov function.
        x0: equilibrium point.
        actual_reachable_set_states: Pre-computed States in real reachable set. 
    """
    x1_vals = np.linspace(-2, 2, 400)
    x2_vals = np.linspace(-2, 2, 400)
    X1r, X2r = np.meshgrid(x1_vals, x2_vals)
    Zr = np.zeros_like(X1r)

    for i in range(X1r.shape[0]):
        for j in range(X1r.shape[1]):
            x_vec = np.array([X1r[i, j], X2r[i, j]])
            x_vec_higher = np.copy(x_vec)
            x_vec_list = [x_vec]
            for _ in range(1, c):
                x_vec_higher = np.kron(x_vec, x_vec_higher)
                x_vec_list.append(x_vec_higher)
            x_tilda = np.concatenate(x_vec_list)
            Zr[i, j] = x_tilda.T @ P_value @ x_tilda

    # Calculate level set passing through x0
    # Calculate xi_0 using Abdelraouf's formulation.
    xi_list = [np.copy(x0)]
    for i in range(1, c):
        xi_list.append(np.kron(x0, xi_list[-1]))
    xc_0 = np.concatenate(xi_list)
    level_set = xc_0.T @ P_value @ xc_0

    # Find contour for the level set and return it
    contour = plt.contour(X1r, X2r, Zr, 
                        levels=[level_set], 
                        colors=[colors[c_values.index(c)]], )
    if actual_reachable_set_states is not None:
        plt.fill(
            actual_reachable_set_states[0, :], 
            actual_reachable_set_states[1, :], 
            color=(240/255, 230/255, 180/255), 
            label='Reachable Set'
            )