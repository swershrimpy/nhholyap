"""
Fault Diagnosis for Unitree Go2 Robot

Supports:
1. Actuator fault: Diminished turn rate (yaw rate effectiveness factor)
2. Sensor fault: IMU drift (affine transform on yaw: a*yaw + b)

Output: Estimated fault-induced errors [px, py, yaw]
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class FaultParameters:
    """Container for fault parameters"""
    # Actuator fault: actual_yaw_rate = alpha * commanded_yaw_rate
    actuator_effectiveness: float = 1.0  # alpha in [0, 1], 1.0 = no fault

    # Sensor fault: measured_yaw = a * true_yaw + b
    sensor_scale: float = 1.0  # a, typically close to 1.0
    sensor_bias: float = 0.0   # b in radians

    def __str__(self):
        return (f"Actuator effectiveness: {self.actuator_effectiveness:.3f}\n"
                f"Sensor scale: {self.sensor_scale:.3f}\n"
                f"Sensor bias: {self.sensor_bias:.3f} rad")


class FaultDiagnosisGo2:
    """
    Fault diagnosis for Unitree Go2 using observation residuals
    """

    def __init__(self, nominal_traj, nominal_input, C_matrix, state_offset, dt=0.5):
        """
        Args:
            nominal_traj: Nominal state trajectory (nx, N+1)
            nominal_input: Nominal control inputs (nu, N)
            C_matrix: Observation matrix (ny, nx)
            state_offset: State offset for observation model
            dt: Time step in seconds
        """
        self.nominal_traj = nominal_traj
        self.nominal_input = nominal_input
        self.C = C_matrix
        self.state_offset = state_offset
        self.dt = dt

        self.nx = nominal_traj.shape[0]  # State dimension (3)
        self.nu = nominal_input.shape[0]  # Control dimension (3)
        self.ny = C_matrix.shape[0]       # Observation dimension (3)
        self.N = nominal_input.shape[1]   # Horizon length

    def simulate_faulty_trajectory(self, fault_params: FaultParameters) -> np.ndarray:
        """
        Simulate trajectory under fault conditions

        Args:
            fault_params: Fault parameters

        Returns:
            Faulty state trajectory (nx, N+1)
        """
        x_faulty = np.zeros((self.nx, self.N + 1))
        x_faulty[:, 0] = self.nominal_traj[:, 0].copy()

        alpha = fault_params.actuator_effectiveness

        for t in range(self.N):
            # Current state
            x = x_faulty[0, t]
            y = x_faulty[1, t]
            theta = x_faulty[2, t]

            # Commanded control (from nominal plan)
            vx_cmd = self.nominal_input[0, t]
            vy_cmd = self.nominal_input[1, t]
            omega_cmd = self.nominal_input[2, t]

            # Actuator fault: reduced yaw rate effectiveness
            omega_actual = alpha * omega_cmd

            # Unicycle dynamics with faulty actuator
            x_next = x + self.dt * (vx_cmd * np.cos(theta) - vy_cmd * np.sin(theta))
            y_next = y + self.dt * (vx_cmd * np.sin(theta) + vy_cmd * np.cos(theta))
            theta_next = theta + self.dt * omega_actual

            x_faulty[:, t + 1] = [x_next, y_next, theta_next]

        return x_faulty

    def apply_sensor_fault(self, true_traj: np.ndarray, fault_params: FaultParameters) -> np.ndarray:
        """
        Apply sensor fault to true trajectory

        Args:
            true_traj: True state trajectory (nx, N+1)
            fault_params: Fault parameters

        Returns:
            Measured trajectory with sensor fault (nx, N+1)
        """
        measured_traj = true_traj.copy()

        # Apply affine transform to yaw measurements
        a = fault_params.sensor_scale
        b = fault_params.sensor_bias

        measured_traj[2, :] = a * true_traj[2, :] + b

        return measured_traj

    def compute_observations(self, state_traj: np.ndarray) -> np.ndarray:
        """
        Compute observations from state trajectory

        Args:
            state_traj: State trajectory (nx, N+1)

        Returns:
            Observations (ny, N+1)
        """
        observations = np.zeros((self.ny, self.N + 1))

        for t in range(self.N + 1):
            state_with_offset = state_traj[:, t] + self.state_offset
            observations[:, t] = self.C @ state_with_offset

        return observations

    def observation_residual(self, fault_params: FaultParameters,
                            measured_observations: np.ndarray) -> np.ndarray:
        """
        Compute observation residuals given fault parameters

        Args:
            fault_params: Fault parameters to evaluate
            measured_observations: Actual observations from robot (ny, N+1)

        Returns:
            Residual vector (flattened)
        """
        # Simulate faulty trajectory (with actuator fault)
        faulty_traj = self.simulate_faulty_trajectory(fault_params)

        # Apply sensor fault to get measured trajectory
        measured_traj = self.apply_sensor_fault(faulty_traj, fault_params)

        # Compute expected observations
        expected_obs = self.compute_observations(measured_traj)

        # Compute residuals
        residuals = measured_observations - expected_obs

        return residuals.flatten()

    def estimate_faults(self, measured_observations: np.ndarray,
                       method='least_squares',
                       initial_guess: Optional[FaultParameters] = None) -> Tuple[FaultParameters, dict]:
        """
        Estimate fault parameters from observations

        Args:
            measured_observations: Observed measurements (ny, N+1)
            method: 'least_squares' or 'minimize'
            initial_guess: Initial fault parameters (if None, use no-fault)

        Returns:
            Estimated fault parameters and optimization info
        """
        if initial_guess is None:
            initial_guess = FaultParameters()

        # Pack parameters into vector: [alpha, a, b]
        x0 = np.array([
            initial_guess.actuator_effectiveness,
            initial_guess.sensor_scale,
            initial_guess.sensor_bias
        ])

        def residual_function(x):
            fault_params = FaultParameters(
                actuator_effectiveness=x[0],
                sensor_scale=x[1],
                sensor_bias=x[2]
            )
            return self.observation_residual(fault_params, measured_observations)

        def cost_function(x):
            residuals = residual_function(x)
            return 0.5 * np.sum(residuals**2)

        # Bounds: alpha in [0, 1], a in [0.5, 1.5], b in [-pi, pi]
        bounds_lower = [0.0, 0.5, -np.pi]
        bounds_upper = [1.0, 1.5, np.pi]

        if method == 'least_squares':
            result = least_squares(
                residual_function,
                x0,
                bounds=(bounds_lower, bounds_upper),
                verbose=0,
                max_nfev=1000
            )
            x_opt = result.x
            success = result.success
            cost = result.cost

        elif method == 'minimize':
            bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(3)]
            result = minimize(
                cost_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            x_opt = result.x
            success = result.success
            cost = result.fun
        else:
            raise ValueError(f"Unknown method: {method}")

        estimated_faults = FaultParameters(
            actuator_effectiveness=x_opt[0],
            sensor_scale=x_opt[1],
            sensor_bias=x_opt[2]
        )

        info = {
            'success': success,
            'cost': cost,
            'residual_norm': np.linalg.norm(residual_function(x_opt)),
            'iterations': result.nfev if method == 'least_squares' else result.nit
        }

        return estimated_faults, info

    def compute_fault_induced_errors(self, fault_params: FaultParameters) -> np.ndarray:
        """
        Compute position and heading errors induced by faults

        Args:
            fault_params: Estimated fault parameters

        Returns:
            Error vector [px, py, yaw] at final time
        """
        # Simulate trajectory with faults
        faulty_traj = self.simulate_faulty_trajectory(fault_params)

        # Compute errors relative to nominal trajectory
        final_position_error_x = faulty_traj[0, -1] - self.nominal_traj[0, -1]
        final_position_error_y = faulty_traj[1, -1] - self.nominal_traj[1, -1]
        final_heading_error = faulty_traj[2, -1] - self.nominal_traj[2, -1]

        # Wrap heading error to [-pi, pi]
        final_heading_error = np.arctan2(np.sin(final_heading_error),
                                         np.cos(final_heading_error))

        return np.array([final_position_error_x, final_position_error_y, final_heading_error])

    def diagnose(self, measured_observations: np.ndarray,
                verbose: bool = True) -> Tuple[np.ndarray, FaultParameters]:
        """
        Full fault diagnosis pipeline

        Args:
            measured_observations: Actual observations from robot (ny, N+1)
            verbose: Whether to print diagnostic information

        Returns:
            Fault-induced errors [px, py, yaw] and estimated fault parameters
        """
        if verbose:
            print("=" * 60)
            print("FAULT DIAGNOSIS FOR UNITREE GO2")
            print("=" * 60)

        # Estimate fault parameters
        estimated_faults, info = self.estimate_faults(measured_observations)

        if verbose:
            print("\nEstimated Fault Parameters:")
            print("-" * 60)
            print(estimated_faults)
            print(f"\nOptimization Info:")
            print(f"  Success: {info['success']}")
            print(f"  Final cost: {info['cost']:.6f}")
            print(f"  Residual norm: {info['residual_norm']:.6f}")
            print(f"  Iterations: {info['iterations']}")

        # Compute fault-induced errors
        errors = self.compute_fault_induced_errors(estimated_faults)

        if verbose:
            print(f"\nFault-Induced Errors:")
            print("-" * 60)
            print(f"  Position error (x): {errors[0]:+.4f} m")
            print(f"  Position error (y): {errors[1]:+.4f} m")
            print(f"  Heading error:      {errors[2]:+.4f} rad ({np.degrees(errors[2]):+.2f} deg)")
            print(f"  Total position error: {np.linalg.norm(errors[:2]):.4f} m")

            # Fault interpretation
            print(f"\nFault Interpretation:")
            print("-" * 60)

            if estimated_faults.actuator_effectiveness < 0.95:
                reduction = (1.0 - estimated_faults.actuator_effectiveness) * 100
                print(f"  ⚠ ACTUATOR FAULT DETECTED:")
                print(f"    Turn rate reduced by {reduction:.1f}%")
            else:
                print(f"  ✓ No significant actuator fault")

            if abs(estimated_faults.sensor_scale - 1.0) > 0.05 or abs(estimated_faults.sensor_bias) > 0.05:
                print(f"  ⚠ SENSOR FAULT DETECTED:")
                if abs(estimated_faults.sensor_scale - 1.0) > 0.05:
                    print(f"    IMU scale error: {(estimated_faults.sensor_scale - 1.0) * 100:+.2f}%")
                if abs(estimated_faults.sensor_bias) > 0.05:
                    print(f"    IMU bias: {estimated_faults.sensor_bias:+.4f} rad ({np.degrees(estimated_faults.sensor_bias):+.2f} deg)")
            else:
                print(f"  ✓ No significant sensor fault")

            print("=" * 60)

        return errors, estimated_faults


class OnlineFaultDiagnosis:
    """
    Online/streaming fault diagnosis using sliding window
    """

    def __init__(self, nominal_traj, nominal_input, C_matrix, state_offset,
                 window_size=10, dt=0.5):
        """
        Args:
            window_size: Number of time steps in sliding window
        """
        self.window_size = window_size
        self.dt = dt

        # Store full trajectory for reference
        self.full_nominal_traj = nominal_traj
        self.full_nominal_input = nominal_input
        self.C = C_matrix
        self.state_offset = state_offset

        # Observation history
        self.observation_history = []
        self.time_history = []

        # Fault estimate history
        self.fault_history = []
        self.error_history = []

    def add_observation(self, observation: np.ndarray, time_step: int):
        """
        Add new observation and update fault estimate

        Args:
            observation: New observation (ny,)
            time_step: Current time step

        Returns:
            Current fault estimate [px, py, yaw] or None if insufficient data
        """
        self.observation_history.append(observation)
        self.time_history.append(time_step)

        # Keep only recent window
        if len(self.observation_history) > self.window_size:
            self.observation_history.pop(0)
            self.time_history.pop(0)

        # Need at least 5 observations for reliable estimation
        if len(self.observation_history) < 5:
            return None, None

        # Extract window of nominal trajectory
        start_idx = self.time_history[0]
        end_idx = self.time_history[-1] + 1

        window_nominal_traj = self.full_nominal_traj[:, start_idx:end_idx+1]
        window_nominal_input = self.full_nominal_input[:, start_idx:end_idx]

        # Stack observations
        window_observations = np.column_stack(self.observation_history)

        # Create fault diagnosis instance for window
        fd = FaultDiagnosisGo2(
            window_nominal_traj,
            window_nominal_input,
            self.C,
            self.state_offset,
            self.dt
        )

        # Estimate faults (use previous estimate as initial guess if available)
        initial_guess = self.fault_history[-1] if self.fault_history else None

        try:
            errors, faults = fd.diagnose(window_observations, verbose=False)

            self.fault_history.append(faults)
            self.error_history.append(errors)

            return errors, faults
        except Exception as e:
            warnings.warn(f"Fault diagnosis failed: {e}")
            return None, None

    def get_fault_trend(self, n_recent=5) -> Optional[dict]:
        """
        Get trend of recent fault estimates

        Returns:
            Dictionary with mean and std of recent estimates
        """
        if len(self.fault_history) < n_recent:
            return None

        recent_faults = self.fault_history[-n_recent:]

        alphas = [f.actuator_effectiveness for f in recent_faults]
        scales = [f.sensor_scale for f in recent_faults]
        biases = [f.sensor_bias for f in recent_faults]

        return {
            'actuator_effectiveness': {
                'mean': np.mean(alphas),
                'std': np.std(alphas)
            },
            'sensor_scale': {
                'mean': np.mean(scales),
                'std': np.std(scales)
            },
            'sensor_bias': {
                'mean': np.mean(biases),
                'std': np.std(biases)
            }
        }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Fault Diagnosis Module\n")

    # Create synthetic nominal trajectory
    N = 20
    dt = 0.5

    # Simple circular trajectory
    nominal_traj = np.zeros((3, N+1))
    nominal_input = np.zeros((3, N))

    for t in range(N+1):
        nominal_traj[0, t] = np.cos(0.1 * t)  # x
        nominal_traj[1, t] = np.sin(0.1 * t)  # y
        nominal_traj[2, t] = 0.1 * t           # theta

        if t < N:
            nominal_input[0, t] = 0.2   # vx
            nominal_input[1, t] = 0.0   # vy
            nominal_input[2, t] = 0.2   # omega

    # Observation matrix
    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])

    state_offset = np.array([-0.46, 0.34, 0.0])

    # Create fault diagnosis instance
    fd = FaultDiagnosisGo2(nominal_traj, nominal_input, C, state_offset, dt)

    # Simulate fault scenario
    print("Simulating fault scenario...")
    true_faults = FaultParameters(
        actuator_effectiveness=0.7,  # 30% reduction in turn rate
        sensor_scale=1.05,            # 5% scale error
        sensor_bias=0.1               # ~5.7 degree bias
    )

    print(f"\nTrue faults:")
    print(true_faults)

    # Generate faulty trajectory and observations
    faulty_traj = fd.simulate_faulty_trajectory(true_faults)
    measured_traj = fd.apply_sensor_fault(faulty_traj, true_faults)
    measured_obs = fd.compute_observations(measured_traj)

    # Add some noise
    measured_obs += 0.01 * np.random.randn(*measured_obs.shape)

    # Diagnose faults
    errors, estimated_faults = fd.diagnose(measured_obs, verbose=True)

    print(f"\nTrue errors: {fd.compute_fault_induced_errors(true_faults)}")
    print(f"Estimated errors: {errors}")
