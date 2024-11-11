import scipy
import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp

def calculate_lyapunov_P_matrix_abdelraouf(c, x0, A_list):
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
    # return
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
    problem.solve(solver=cp.CLARABEL)#, eps_abs=1e-12, eps_rel=1e-12)
    if P.value is None:
        raise ValueError("No solution found")

    P_value = P.value   
    return P_value