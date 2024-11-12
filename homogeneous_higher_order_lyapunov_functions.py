import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def calculate_lyapunov_P_matrix(c, A_list):
    """
    Calculates P matrix for homogeneous HO Lyapunov function according to Matthew Abate's paper.
    Works for arbitrary dimension - SLOW. 2D - FAST, with dimensionality reduction matrix implemented.
    Inputs:
        c: order of Lyapunov function.
        A_list: list of A matrices in set M.

    Returns:
        P_value: P matrix for nonhomogeneous HO Lyapunov function.
    """
    state_length = A_list[0].shape[0]
    use_W_flag = state_length == 2
    if use_W_flag:
        W = np.eye(state_length)
        for i in range(1, c):
            W = np.block([[W, np.zeros((2**i, 1))], [np.zeros((2**i, 1)), W]])

    # Construct all B matrices
    Ac_list = [np.copy(A_list[i]) for i in range(len(A_list))]
    for i in range(1, c):
        Ac_list = [np.kron(np.eye(state_length), Ac_list[_]) + np.kron(A_list[_], np.eye(state_length**i))
                   for _ in range(len(Ac_list))]
    if use_W_flag:
        Bc_list = [np.linalg.pinv(W) @ Ac @ W for Ac in Ac_list]

        # Solve for P using CVXPY
        P = cp.Variable((c + 1, c + 1), symmetric=True)
        constraints = [
            Bc.T @ P + P @ Bc << 0 for Bc in Bc_list]
        constraints.append(
            P >> np.eye(c + 1)
        )
       
    else:
        P = cp.Variable((state_length ** c, state_length ** c), symmetric=True)
        constraints = [
            Ac.T @ P + P @ Ac << 0 for Ac in Ac_list]
        constraints.append(
            P >> np.eye(state_length ** c)
        )
    objective = cp.Minimize(P[0, 0])
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if P.value is None:
        raise ValueError("No solution found")

    P_value = P.value   
    return P_value


def meta_lyapunov_with_list_input(c, 
                                  x0,
                                  A_list,
                                  c_values,
                                  colors = [
        (44/255, 123/255, 182/255),
        (171/255, 217/255, 233/255),
        (253/255, 174/255, 97/255),
        (215/255, 25/255, 28/255)
    ]):
    """
    Compute and return the level set for the Meta Lyapunov function of order 2*c.
    TODO: rewrite. This is probably not necessary - a separate P-matrix calculation function
    and a plotting function should suffice.
    """
    # Construct W matrix
    assert len(A_list) > 0, "List of A matrices must not be empty."

    state_length = A_list[0].shape[0]
    use_W_flag = state_length == 2
    if use_W_flag:
        W = np.eye(state_length)
        for i in range(1, c):
            W = np.block([[W, np.zeros((2**i, 1))], [np.zeros((2**i, 1)), W]])

    # Construct all B matrices
    Ac_list = [np.copy(A_list[i]) for i in range(len(A_list))]
    for i in range(1, c):
        Ac_list = [np.kron(np.eye(state_length), Ac_list[_]) + np.kron(A_list[_], np.eye(state_length**i))
                   for _ in range(len(Ac_list))]
        # Ac2 = np.kron(np.eye(state_length), Ac2) + np.kron(A2, np.eye(state_length**i))
    print(Ac_list[0].shape)
    if use_W_flag:
        print(W.shape)
        Bc_list = [np.linalg.pinv(W) @ Ac @ W for Ac in Ac_list]
        # Bc2 = np.linalg.pinv(W) @ Ac2 @ W

        # Solve for P using CVXPY
        P = cp.Variable((c + 1, c + 1), symmetric=True)
        constraints = [
            Bc.T @ P + P @ Bc << 0 for Bc in Bc_list]
        constraints.append(
            P >> np.eye(c + 1)
        )
        objective = cp.Minimize(P[0, 0])
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)#, eps_abs=1e-12, eps_rel=1e-12)

        # if problem.status in ["infeasible", "unbounded"]:
        #     raise ValueError("No solution found")
        if P.value is None:
            print("No solution found")
            raise ValueError(f"{state_length*c}th order: no solution found")

        P_value = P.value
        print(f"P_value for {c}th order is: ")
        print(P_value)
        # Discretize the domain to find the level set
        x1_vals = np.linspace(-2, 2, 400)
        x2_vals = np.linspace(-2, 2, 400)
        X1r, X2r = np.meshgrid(x1_vals, x2_vals)
        Zr = np.zeros_like(X1r)

        for i in range(X1r.shape[0]):
            for j in range(X1r.shape[1]):
                x_vec = np.array([X1r[i, j]**(c - g) * X2r[i, j]**g for g in range(c + 1)])
                Zr[i, j] = x_vec.T @ P_value @ x_vec

        # Calculate level set passing through x0
        x0_vec = np.array([x0[0]**(c - g) * x0[1]**g for g in range(c + 1)])
        level_set = x0_vec.T @ P_value @ x0_vec

        # Find contour for the level set and return it
        contour = plt.contour(X1r, X2r, Zr, 
                            levels=[level_set], 
                            colors=[colors[c_values.index(c)]], )
    else:
        P = cp.Variable((state_length ** c, state_length ** c), symmetric=True)
        constraints = [
            Ac.T @ P + P @ Ac << 0 for Ac in Ac_list]
        constraints.append(
            P >> np.eye(state_length ** c)
        )
        objective = cp.Minimize(P[0, 0])
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)#, eps_abs=1e-12, eps_rel=1e-12)
        if P.value is None:
            raise ValueError("No solution found")

        P_value = P.value
        # Discretize the domain to find the level set
        x1_vals = np.linspace(-2, 2, 400)
        x2_vals = np.linspace(-2, 2, 400)
        X1r, X2r = np.meshgrid(x1_vals, x2_vals)
        Zr = np.zeros_like(X1r)

        for i in range(X1r.shape[0]):
            for j in range(X1r.shape[1]):
                x_vec = np.array([X1r[i, j]**(c - g) * X2r[i, j]**g for g in range(c + 1)])
                Zr[i, j] = x_vec.T @ P_value @ x_vec

        # Calculate level set passing through x0
        x0_vec = np.array([x0[0]**(c - g) * x0[1]**g for g in range(c + 1)])
        level_set = x0_vec.T @ P_value @ x0_vec

        # Find contour for the level set and return it
        contour = plt.contour(X1r, X2r, Zr, 
                            levels=[level_set], 
                            colors=[colors[c_values.index(c)]], )
    
    return contour


def plot_reachable_set_from_hierarchical_lyap_func(A_list, x0, c_values=[1, 5, 8, 9], actual_reachable_set_states=None):
    """
    Plots reachable set resulting from Hierarchical Lyapunov Function.
    TODO: Rewrite to separate P matrix calculation and plotting.
    """
    state_length = x0.size

    # Define colors for plotting
    colors = [
        (44/255, 123/255, 182/255),
        (171/255, 217/255, 233/255),
        (253/255, 174/255, 97/255),
        (215/255, 25/255, 28/255)
    ]

    # Plot setup
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

    # Plot level sets for each value in c_values
    for i, c in enumerate(c_values):
        # try:
        contour = meta_lyapunov_with_list_input(c, A_list=A_list, x0=x0, c_values=c_values)
        label = f"{state_length*c}th order" if c != 1 else f"{state_length*c}nd order"
        plt.clabel(contour, fmt=label)
        # except ValueError as ve:
        #     print(ve)
            # print(f"{state_length*c}th order failed" if c != 1 else f"{state_length*c}nd order failed")

    # Plot initial condition x0 = [1, 0]
    # plt.scatter(x0[0], x0[1], s=80, c='green', marker='d', label='Initial State')
    if actual_reachable_set_states is not None:
        plt.fill(
            actual_reachable_set_states[0, :], 
            actual_reachable_set_states[1, :], 
            color=(240/255, 230/255, 180/255), 
            label='Reachable Set'
            )

    plt.legend(loc='lower right', fontsize=12)
    plt.show()