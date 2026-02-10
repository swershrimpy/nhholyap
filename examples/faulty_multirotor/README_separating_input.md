# Separating Input Optimizer

A Python module for computing optimal "separating inputs" using gradient descent and JAX automatic differentiation.

## What is a Separating Input?

A **separating input** is a control signal that minimizes the overlap between interval reachable sets for different fault scenarios. By maximizing the separation between reachable sets, these inputs make it easier to distinguish between different faults through observation of the system's behavior.

### Key Concepts

- **Fault Scenarios**: Different system parameter values representing various fault conditions (e.g., actuator degradation, sensor drift)
- **Interval Propagation**: Forward simulation of uncertain initial states and parameters, resulting in interval-valued reachable sets
- **Overlap Minimization**: Finding control inputs that minimize the volume of intersection between reachable sets from different fault scenarios

## Features

- **Automatic Differentiation**: Uses JAX for efficient gradient computation
- **JIT Compilation**: Fast execution through JAX's just-in-time compilation
- **Multi-Start Optimization**: Global optimization with multiple random initializations
- **Flexible Configuration**: Support for multiple fault scenarios, custom state slicing, and various propagation parameters
- **Comprehensive Testing**: Full unit test suite with 24 tests covering edge cases

## Installation

This module requires:
- JAX
- immrax (interval-based reachability library)
- NumPy

Install using conda:
```bash
conda activate immrax
```

## Quick Start

```python
import jax.numpy as jnp
import immrax as irx
from faulty_planar_multirotor import FaultyPlanarMultirotor
from separating_input_optimizer import SeparatingInputOptimizer

# 1. Define system
sys = FaultyPlanarMultirotor()

# 2. Initial state interval
x0 = irx.icentpert(
    jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # Center
    jnp.array([0.1, 0.1, 0.05, 0.05, 0.05]),  # Perturbation
)

# 3. Disturbance interval
w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

# 4. Fault scenarios (nominal and 50% thrust loss)
p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
p_fault = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

# 5. Create optimizer
optimizer = SeparatingInputOptimizer(
    system=sys,
    fault_parameters=[p_nominal, p_fault],
    x0_interval=x0,
    w_interval=w,
    dt=0.02,
    num_steps=20,
    state_slice=slice(0, 2),  # Only consider position
)

# 6. Optimize
u_opt, loss = optimizer.optimize(
    learning_rate=1e-1,
    num_iterations=100,
)

print(f"Optimal control: {u_opt}")
print(f"Overlap: {loss}")
```

## API Reference

### SeparatingInputOptimizer

Main optimizer class for finding separating inputs.

**Constructor Parameters:**
- `system`: The dynamical system (e.g., `FaultyPlanarMultirotor`)
- `fault_parameters`: List of parameter intervals for each fault scenario
- `x0_interval`: Initial state interval
- `w_interval`: Disturbance interval
- `dt`: Time step for propagation
- `num_steps`: Number of propagation steps
- `state_slice`: Optional slice to select which states to consider (default: all states)

**Methods:**

#### `optimize(u_initial=None, learning_rate=1e-1, num_iterations=100, verbose=False)`

Run gradient descent optimization.

- **Returns**: `(u_optimal, final_loss)` tuple
- **Parameters**:
  - `u_initial`: Initial control guess (random if None)
  - `learning_rate`: Gradient descent step size
  - `num_iterations`: Number of optimization steps
  - `verbose`: Print progress if True

#### `evaluate(u)`

Evaluate a control input.

- **Returns**: `(loss, list_of_intervals)` tuple
- **Parameters**:
  - `u`: Control input to evaluate

### optimize_separating_input_multistart

Multi-start optimization for global optimum search.

**Parameters:**
- All `SeparatingInputOptimizer` constructor parameters, plus:
- `num_restarts`: Number of random initializations
- `learning_rate`: Gradient descent step size
- `num_iterations`: GD iterations per restart
- `random_key`: Random seed

**Returns**: `(best_u, best_loss)` tuple

## Examples

### Example 1: Two Fault Scenarios

```python
# Nominal vs. 50% thrust loss
p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
p_fault = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

optimizer = SeparatingInputOptimizer(
    system=sys,
    fault_parameters=[p_nominal, p_fault],
    x0_interval=x0,
    w_interval=w,
    dt=0.02,
    num_steps=20,
    state_slice=slice(0, 2),  # Position only
)

u_opt, loss = optimizer.optimize(num_iterations=100)
```

### Example 2: Three Fault Scenarios with Multi-Start

```python
from separating_input_optimizer import optimize_separating_input_multistart

p_100 = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
p_70 = irx.icentpert(jnp.array([0.7, 1.0]), jnp.zeros(2))
p_40 = irx.icentpert(jnp.array([0.4, 1.0]), jnp.zeros(2))

u_opt, loss = optimize_separating_input_multistart(
    system=sys,
    fault_parameters=[p_100, p_70, p_40],
    x0_interval=x0,
    w_interval=w,
    dt=0.02,
    num_steps=15,
    num_restarts=5,  # Try 5 random initializations
    num_iterations=100,
    state_slice=slice(0, 2),
)
```

### Example 3: Uncertain Fault Parameters

```python
# Fault with parameter uncertainty
p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
p_fault_uncertain = irx.icentpert(
    jnp.array([0.5, 1.0]),  # Center
    jnp.array([0.2, 0.0]),  # Uncertainty: thrust in [0.3, 0.7]
)

optimizer = SeparatingInputOptimizer(
    system=sys,
    fault_parameters=[p_nominal, p_fault_uncertain],
    x0_interval=x0,
    w_interval=w,
    dt=0.02,
    num_steps=20,
)

u_opt, loss = optimizer.optimize()
```

## Running Tests

The module includes comprehensive unit tests:

```bash
# Run all tests
conda activate immrax
cd examples/faulty_multirotor
python -m pytest tests/test_separating_input_optimizer.py -v

# Run specific test class
python -m pytest tests/test_separating_input_optimizer.py::TestOverlapFunctions -v

# Run with coverage
python -m pytest tests/test_separating_input_optimizer.py --cov=separating_input_optimizer
```

All 24 tests should pass:
- 5 overlap function tests
- 5 pairwise overlap sum tests
- 2 interval propagation tests
- 7 optimizer tests
- 2 multi-start tests
- 1 three-scenario test
- 2 edge case tests

## Running the Demo

A comprehensive demo script is provided:

```bash
conda activate immrax
python demo_separating_input.py
```

The demo includes three scenarios:
1. Two fault scenarios (nominal vs. 50% thrust loss)
2. Three fault scenarios (100%, 70%, 40% thrust)
3. Uncertain fault parameters

## Implementation Details

### Gradient Descent

The optimizer uses vanilla gradient descent with a fixed learning rate:
```
u_{k+1} = u_k - α * ∇_u L(u_k)
```

where:
- `α` is the learning rate
- `L(u)` is the overlap loss function
- `∇_u L` is computed via JAX automatic differentiation

### Loss Function

The loss is the sum of pairwise overlaps:
```
L(u) = Σ_{i<j} overlap(R_i(u), R_j(u))
```

where `R_i(u)` is the reachable set under fault scenario `i` with control `u`.

### Interval Propagation

Forward Euler method:
```
x_{k+1} = x_k + dt * f(x_k, u, w, p)
```

Extended to intervals using natural interval extensions with `immrax`.

## Limitations and Future Work

### Current Limitations
- Only constant control inputs (no time-varying sequences in base optimizer)
- Forward Euler propagation (less accurate than higher-order methods)
- Local optimization (gradient descent can get stuck in local minima)
- No constraint handling on control inputs

### Potential Extensions
- Time-varying control sequences (see `calculate_optimal_open_loop_u.py` for an example)
- Higher-order ODE solvers for propagation
- Constraint handling (e.g., box constraints on control)
- Adaptive learning rates
- Alternative optimization algorithms (L-BFGS, Adam, etc.)
- Parallelization across multiple scenarios

## References

This implementation is based on concepts from:
- Interval reachability analysis
- Active fault diagnosis
- Set-based estimation
- Gradient-based optimal control

## License

Part of the nhholyap (output feedback) research codebase.

## Contact

For questions or issues, please consult the main repository documentation.
