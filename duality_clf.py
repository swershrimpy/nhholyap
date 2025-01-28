import cvxpy as cp
import numpy as np

def find_lyapunov_and_gain(A, B, mu, x0):
    """
    Solve for the Lyapunov function V(x) = x^T Q x and control gain K = Y Q^{-1}
    for the system \dot{x} = Ax + Bu with constraints.

    Args:
        A (numpy.ndarray): State matrix (n x n).
        B (numpy.ndarray): Input matrix (n x m).
        mu (float): Control constraint.
        x0 (numpy.ndarray): Initial condition (n x 1).

    Returns:
        dict: A dictionary containing optimal Q, Y, and K if the problem is solvable.
    """
    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Input dimension

    # Decision variables
    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    # Constraints
    constraints = []

    # 1. Q > 0 (Positive definiteness of Q)
    constraints.append(Q >> 0)

    # 2. Lyapunov inequality: AQ + QA^T + BY + Y^T B^T < 0
    constraints.append(A @ Q + Q @ A.T + B @ Y + Y.T @ B.T << 0)

    # 3. Initial condition constraint: [1 x(0)^T; x(0) Q] \geq 0
    # X0_block = cp.bmat([
        # [cp.Constant(1), x0.T],
        # [x0, Q]
    # ])
    X0_block = cp.bmat([[cp.Constant(np.ones((1, 1))), x0.T ], [x0, Q]])
    constraints.append(X0_block >> 0)

    # 4. Input constraint: [Q Y^T; Y \mu^2 I] \geq 0
    input_block = cp.bmat([
        [Q, Y.T],
        [Y, mu**2 * np.eye(m)]
    ])
    constraints.append(input_block >> 0)

    # Objective function: Minimize trace(Q) (optional, can be replaced)
    objective = cp.Maximize(cp.trace(Q))

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Results
    if problem.status == cp.OPTIMAL:
        Q_opt = Q.value
        Y_opt = Y.value
        K_opt = Y_opt @ np.linalg.inv(Q_opt)  # Compute the control gain K

        return {
            "Q": Q_opt,
            "Y": Y_opt,
            "K": K_opt
        }
    else:
        return {
            "status": problem.status,
            "message": "Problem is not solvable."
        }

# Example usage (replace A, B, mu, x0 with specific values)
if __name__ == "__main__":
    A = np.array([1]).reshape((1, 1))  # Example state matrix
    B = np.array([1]).reshape((1, 1))         # Example input matrix
    mu = 3.0                         # Example control constraint
    x0 = np.array([3]).reshape((1, 1))        # Example initial condition

    result = find_lyapunov_and_gain(A, B, mu, x0)

    if "Q" in result:
        print("Optimal Q:")
        print(result["Q"])
        print("Optimal Y:")
        print(result["Y"])
        print("Optimal control gain K:")
        print(result["K"])
    else:
        print("Status:", result["status"])
        print("Message:", result["message"])
