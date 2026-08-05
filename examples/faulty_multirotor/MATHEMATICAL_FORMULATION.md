# Mathematical Formulation: Separating Input Optimization

This document provides the complete mathematical formulation of all steps in the separating input optimization demonstration.

---

## 1. System Dynamics

### 1.1 Planar Multirotor Dynamics

The faulty planar multirotor is described by the following continuous-time nonlinear system:

**State vector:**
$$\mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \\ \theta \end{bmatrix} \in \mathbb{R}^5$$

where:
- $p_x, p_y$ are the 2D position coordinates (m)
- $v_x, v_y$ are the 2D velocity components (m/s)
- $\theta$ is the pitch angle (rad)

**Control input:**
$$\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} \in \mathbb{R}^2$$

where:
- $u_1$ is the total thrust magnitude (m/s²)
- $u_2$ is the angular acceleration command (rad/s²)

**Disturbance:**
$$\mathbf{w} = \begin{bmatrix} w_1 \end{bmatrix} \in \mathbb{R}^1$$

where $w_1$ is an additive disturbance on the $v_x$ dynamics.

**Fault parameters:**
$$\mathbf{p} = \begin{bmatrix} p_1 \\ p_2 \end{bmatrix} \in [0,1]^2$$

where:
- $p_1$ is the thrust effectiveness factor ($1.0$ = nominal, $<1.0$ = rotor fault)
- $p_2$ is the angular acceleration effectiveness factor

### 1.2 System Equations

The continuous-time dynamics are given by:

$$\dot{\mathbf{x}} = f(t, \mathbf{x}, \mathbf{u}, \mathbf{w}, \mathbf{p})$$

where:

$$\begin{aligned}
\dot{p}_x &= v_x \\
\dot{p}_y &= v_y \\
\dot{v}_x &= -p_1 u_1 \sin(\theta) + w_1 \\
\dot{v}_y &= p_1 u_1 \cos(\theta) - g \\
\dot{\theta} &= p_2 u_2
\end{aligned}$$

with gravitational constant $g = 9.81$ m/s².

**Nominal case:** $\mathbf{p} = [1.0, 1.0]^T$

**Fault case (50% thrust loss):** $\mathbf{p} = [0.5, 1.0]^T$

---

## 2. Interval Arithmetic and Reachability

### 2.1 Interval Representation

An interval $\mathcal{I} \subset \mathbb{R}^n$ is represented by lower and upper bounds:

$$\mathcal{I} = [\underline{x}, \overline{x}] = \{x \in \mathbb{R}^n : \underline{x}_i \leq x_i \leq \overline{x}_i, \, i=1,\ldots,n\}$$

**Center-perturbation form:**
$$\mathcal{I} = \text{icentpert}(c, r) = [c - r, c + r]$$

where $c \in \mathbb{R}^n$ is the center and $r \in \mathbb{R}_+^n$ is the perturbation radius.

### 2.2 Initial State Interval

The initial state interval is defined as:

$$\mathcal{X}_0 = [\underline{x}_0, \overline{x}_0]$$

**Example 1 (notebook):**
$$\mathcal{X}_0 = \text{icentpert}\left(\begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \end{bmatrix}\right) = \begin{bmatrix} [-0.1, 0.1] \\ [-0.1, 0.1] \\ [-0.1, 0.1] \\ [-0.1, 0.1] \\ [-0.1, 0.1] \end{bmatrix}$$

### 2.3 Interval Embedding

The natural interval embedding extends the state dimension to $2n$:

$$\tilde{x} = \begin{bmatrix} \underline{x} \\ \overline{x} \end{bmatrix} \in \mathbb{R}^{2n}$$

Conversions:
- **Interval to embedding:** $\text{i2ut}(\mathcal{I}) = [\underline{x}^T, \overline{x}^T]^T$
- **Embedding to interval:** $\text{ut2i}(\tilde{x}) = [\tilde{x}_{1:n}, \tilde{x}_{n+1:2n}]$

---

## 3. Interval Propagation

### 3.1 Forward Euler Method

The interval propagation uses the Forward Euler discretization:

$$\mathcal{X}_{k+1} = \mathcal{X}_k + \Delta t \cdot f(0, \mathcal{X}_k, \mathbf{u}, \mathcal{W}, \mathcal{P})$$

In embedding form:

$$\tilde{x}_{k+1} = \tilde{x}_k + \Delta t \cdot f_{\text{emb}}(0, \tilde{x}_k, \mathbf{u}, \mathcal{W}, \mathcal{P})$$

where $f_{\text{emb}}$ is the natural interval extension of $f$.

### 3.2 Multi-Step Propagation

For $N$ time steps with constant control $\mathbf{u}$:

$$\mathcal{X}_N = \Phi_N(\mathcal{X}_0, \mathbf{u}, \mathcal{W}, \mathcal{P})$$

where $\Phi_N$ is the $N$-step flow map, computed iteratively:

$$\mathcal{X}_{k+1} = \Phi_1(\mathcal{X}_k, \mathbf{u}, \mathcal{W}, \mathcal{P}), \quad k = 0, 1, \ldots, N-1$$

**Notebook parameters:**
- Time step: $\Delta t = 0.1$ s (Example 1), $\Delta t = 0.02$ s (Examples 2-3)
- Number of steps: $N = 5$ (Example 1), $N = 10$ (Examples 2-3)
- Total propagation time: $T = N \cdot \Delta t$

### 3.3 Reachable Set for Fault Scenario

For fault scenario $i$ with parameter interval $\mathcal{P}_i$:

$$\mathcal{R}_i(\mathbf{u}) = \Phi_N(\mathcal{X}_0, \mathbf{u}, \mathcal{W}, \mathcal{P}_i)$$

This represents all possible states reachable at time $T$ under control $\mathbf{u}$ when the system operates with fault parameters in $\mathcal{P}_i$.

---

## 4. Overlap Computation

### 4.1 Interval Intersection

The intersection of two intervals $\mathcal{I}_1 = [\underline{x}_1, \overline{x}_1]$ and $\mathcal{I}_2 = [\underline{x}_2, \overline{x}_2]$ is:

$$\mathcal{I}_1 \cap \mathcal{I}_2 = [\max(\underline{x}_1, \underline{x}_2), \min(\overline{x}_1, \overline{x}_2)]$$

If $\max(\underline{x}_1, \underline{x}_2) > \min(\overline{x}_1, \overline{x}_2)$ in any dimension, the intersection is empty: $\mathcal{I}_1 \cap \mathcal{I}_2 = \emptyset$.

### 4.2 Overlap Size (Volume)

The overlap size between two intervals is the volume of their intersection:

$$\text{overlap}(\mathcal{I}_1, \mathcal{I}_2) = \begin{cases}
\prod_{i=1}^{n} (\overline{x}_{\cap,i} - \underline{x}_{\cap,i}) & \text{if } \mathcal{I}_1 \cap \mathcal{I}_2 \neq \emptyset \\
0 & \text{otherwise}
\end{cases}$$

where $[\underline{x}_\cap, \overline{x}_\cap] = \mathcal{I}_1 \cap \mathcal{I}_2$.

**JAX-compatible implementation:**

$$\text{overlap}_{\text{lax}}(\mathcal{I}_1, \mathcal{I}_2) = \text{cond}\left(\neg \text{empty}(\mathcal{I}_1 \cap \mathcal{I}_2), \, \prod_{i} \Delta x_i, \, 0 \right)$$

where $\text{cond}$ is a conditional that works with JAX tracing.

### 4.3 State Slicing

Often, we only consider a subset of states for overlap computation. Given a slice $S \subseteq \{1, \ldots, n\}$:

$$\mathcal{I}_S = \{x_S : x \in \mathcal{I}\}$$

**Notebook example:** Position-only overlap uses $S = \{1, 2\}$ (i.e., $p_x, p_y$ only).

$$\text{overlap}_{\text{pos}}(\mathcal{R}_1, \mathcal{R}_2) = \text{overlap}(\mathcal{R}_{1,[1:2]}, \mathcal{R}_{2,[1:2]})$$

---

## 5. Loss Function

### 5.1 Pairwise Overlap Sum

For $M$ fault scenarios, the total loss is the sum of all pairwise overlaps:

$$L(\mathbf{u}) = \sum_{i=1}^{M-1} \sum_{j=i+1}^{M} \text{overlap}(\mathcal{R}_i(\mathbf{u}), \mathcal{R}_j(\mathbf{u}))$$

**Number of pairs:** $\binom{M}{2} = \frac{M(M-1)}{2}$

**Examples:**
- $M = 2$: One pair $(1,2)$ → $L(\mathbf{u}) = \text{overlap}(\mathcal{R}_1, \mathcal{R}_2)$
- $M = 3$: Three pairs $(1,2), (1,3), (2,3)$ → $L(\mathbf{u}) = \text{overlap}(\mathcal{R}_1, \mathcal{R}_2) + \text{overlap}(\mathcal{R}_1, \mathcal{R}_3) + \text{overlap}(\mathcal{R}_2, \mathcal{R}_3)$

### 5.2 Optimization Objective

The separating input optimization problem is:

$$\mathbf{u}^* = \arg\min_{\mathbf{u} \in \mathcal{U}} L(\mathbf{u})$$

where $\mathcal{U} \subseteq \mathbb{R}^m$ is the set of admissible controls (typically $\mathcal{U} = \mathbb{R}^m$ in the unconstrained case).

**Interpretation:** Find the control that minimizes the total overlap, thereby maximizing the separation between different fault scenarios.

---

## 6. Gradient Descent Optimization

### 6.1 Gradient Computation

The gradient of the loss with respect to the control is:

$$\nabla_{\mathbf{u}} L(\mathbf{u}) = \frac{\partial L}{\partial \mathbf{u}} \in \mathbb{R}^m$$

This gradient is computed using **automatic differentiation** via JAX:

```python
grad_L = jax.grad(loss_fn, argnums=0)
```

The gradient captures how changes in the control affect the overlap:

$$\frac{\partial L}{\partial u_j} = \sum_{i<k} \frac{\partial \text{overlap}(\mathcal{R}_i, \mathcal{R}_k)}{\partial u_j}$$

### 6.2 Gradient Descent Update Rule

Standard gradient descent with fixed step size:

$$\mathbf{u}_{n+1} = \mathbf{u}_n - \alpha \nabla_{\mathbf{u}} L(\mathbf{u}_n)$$

where:
- $\alpha > 0$ is the learning rate (step size)
- $n = 0, 1, \ldots, N_{\text{iter}} - 1$ is the iteration counter

**Notebook parameters:**
- Learning rate: $\alpha = 0.1$
- Number of iterations: $N_{\text{iter}} = 100$ (Example 1), $N_{\text{iter}} = 80$ (Example 2)

### 6.3 Initialization

Initial guess $\mathbf{u}_0$ can be:
1. **Hover input:** $\mathbf{u}_0 = [g, 0]^T = [9.81, 0]^T$
2. **Random:** $\mathbf{u}_0 \sim \mathcal{N}(0, I)$ (multi-start)
3. **User-specified**

### 6.4 Convergence

The optimization terminates after $N_{\text{iter}}$ iterations, returning:

$$\mathbf{u}^* \approx \mathbf{u}_{N_{\text{iter}}}$$

The final loss is:

$$L^* = L(\mathbf{u}^*)$$

---

## 7. Multi-Start Optimization

### 7.1 Motivation

Gradient descent can converge to local minima. Multi-start optimization runs multiple gradient descents from different initializations to find a better solution.

### 7.2 Algorithm

For $K$ random restarts:

1. Generate $K$ initial guesses: $\mathbf{u}_0^{(1)}, \ldots, \mathbf{u}_0^{(K)}$
2. For each $k = 1, \ldots, K$:
   - Run gradient descent from $\mathbf{u}_0^{(k)}$ to obtain $\mathbf{u}^{*(k)}$
   - Compute $L^{(k)} = L(\mathbf{u}^{*(k)})$
3. Select the best:
   $$k^* = \arg\min_{k=1,\ldots,K} L^{(k)}$$
4. Return $\mathbf{u}^* = \mathbf{u}^{*(k^*)}$ and $L^* = L^{(k^*)}$

**Notebook parameters (Example 2):**
- Number of restarts: $K = 3$
- Random seed: 42

### 7.3 Random Initialization

Each restart uses a random initial guess:

$$\mathbf{u}_0^{(k)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

where $\mathcal{N}(\mathbf{0}, \mathbf{I})$ is a standard normal distribution.

---

## 8. Fault Scenarios

### 8.1 Two Fault Scenarios (Example 1)

**Scenario 1 (Nominal):**
$$\mathcal{P}_1 = \text{icentpert}\left(\begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \end{bmatrix}\right) = \begin{bmatrix} [1.0, 1.0] \\ [1.0, 1.0] \end{bmatrix}$$

**Scenario 2 (50% Thrust Loss):**
$$\mathcal{P}_2 = \text{icentpert}\left(\begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \end{bmatrix}\right) = \begin{bmatrix} [0.5, 0.5] \\ [1.0, 1.0] \end{bmatrix}$$

**Loss function:**
$$L(\mathbf{u}) = \text{overlap}\left(\mathcal{R}_1(\mathbf{u})_{[1:2]}, \mathcal{R}_2(\mathbf{u})_{[1:2]}\right)$$

### 8.2 Three Fault Scenarios (Example 2)

**Scenario 1 (100% Thrust):**
$$\mathcal{P}_1 = [1.0, 1.0] \times [1.0, 1.0]$$

**Scenario 2 (70% Thrust):**
$$\mathcal{P}_2 = [0.7, 0.7] \times [1.0, 1.0]$$

**Scenario 3 (40% Thrust):**
$$\mathcal{P}_3 = [0.4, 0.4] \times [1.0, 1.0]$$

**Loss function:**
$$L(\mathbf{u}) = \text{overlap}(\mathcal{R}_1, \mathcal{R}_2) + \text{overlap}(\mathcal{R}_1, \mathcal{R}_3) + \text{overlap}(\mathcal{R}_2, \mathcal{R}_3)$$

where all overlaps are computed on position states only.

### 8.3 Uncertain Fault Parameters (Example 3)

**Scenario 1 (Nominal, Certain):**
$$\mathcal{P}_1 = [1.0, 1.0] \times [1.0, 1.0]$$

**Scenario 2 (Uncertain Thrust, 50% ± 20%):**
$$\mathcal{P}_2 = \text{icentpert}\left(\begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix}, \begin{bmatrix} 0.2 \\ 0 \end{bmatrix}\right) = \begin{bmatrix} [0.3, 0.7] \\ [1.0, 1.0] \end{bmatrix}$$

**Loss function:**
$$L(\mathbf{u}) = \text{overlap}\left(\mathcal{R}_1(\mathbf{u})_{[1:2]}, \mathcal{R}_2(\mathbf{u})_{[1:2]}\right)$$

**Note:** The reachable set $\mathcal{R}_2(\mathbf{u})$ accounts for all possible parameter values in $\mathcal{P}_2 = [0.3, 0.7] \times [1.0, 1.0]$, making it wider than if the parameter were known exactly.

---

## 9. Complete Optimization Pipeline

### 9.1 Inputs

1. **System:** $f(\cdot)$ (planar multirotor dynamics)
2. **Initial state interval:** $\mathcal{X}_0$
3. **Disturbance interval:** $\mathcal{W}$
4. **Fault parameter intervals:** $\{\mathcal{P}_i\}_{i=1}^M$
5. **Time discretization:** $\Delta t$, $N$
6. **State slice:** $S$ (e.g., position only)
7. **Optimization parameters:** $\alpha$, $N_{\text{iter}}$

### 9.2 Algorithm

```
Input: All above parameters
Output: Optimal control u*, final loss L*

1. Initialize: u_0 ← [9.81, 0]  (or random for multi-start)

2. For n = 0 to N_iter - 1:
   a. Propagate all scenarios:
      For i = 1 to M:
         R_i ← Φ_N(X_0, u_n, W, P_i)
         R_i,S ← R_i[S]  (slice to monitored states)

   b. Compute loss:
      L_n ← Σ_{i<j} overlap(R_i,S, R_j,S)

   c. Compute gradient:
      g_n ← ∇_u L(u_n)  (via automatic differentiation)

   d. Update control:
      u_{n+1} ← u_n - α * g_n

3. Return: u* = u_{N_iter}, L* = L(u_{N_iter})
```

### 9.3 JAX JIT Compilation

The entire optimization loop is JIT-compiled for efficiency:

```python
@jax.jit
def gradient_step(i, u_current):
    grad = jax.grad(loss_fn)(u_current)
    return u_current - learning_rate * grad

u_opt = jax.lax.fori_loop(0, N_iter, gradient_step, u_initial)
```

This compiles the loop to optimized XLA code, achieving significant speedup.

---

## 10. Results Interpretation

### 10.1 Overlap Value

The overlap $L(\mathbf{u})$ has units of (meters)$^{|S|}$ where $|S|$ is the dimension of the monitored state space.

**For position-only monitoring** ($S = \{1, 2\}$):
$$[L] = \text{m}^2$$

**Interpretation:**
- $L = 0$: Perfect separation (no overlap)
- $L > 0$: Partial overlap (ambiguous region where faults cannot be distinguished)
- Smaller $L$ → Better fault distinguishability

### 10.2 Optimal Control

The optimal control $\mathbf{u}^*$ represents the best constant input to apply over the time horizon $[0, T]$ to maximize fault separability.

**Example 1 result:**
- Baseline (hover): $\mathbf{u} = [9.81, 0]^T$, $L = 0$
- Optimized: $\mathbf{u}^* = [9.81, 0]^T$, $L^* = 0$
- **Conclusion:** Hover already achieves perfect separation

**Example 2 result:**
- Multi-start optimized: $\mathbf{u}^* = [-1.200, -0.972]^T$, $L^* = 0.168$
- **Conclusion:** Applying downward thrust and negative angular acceleration provides best separation among 3 scenarios

### 10.3 Reachable Sets

The final reachable sets are:

$$\mathcal{R}_i^* = \Phi_N(\mathcal{X}_0, \mathbf{u}^*, \mathcal{W}, \mathcal{P}_i)$$

These can be visualized as rectangles in the $(p_x, p_y)$ plane to assess separation quality.

---

## 11. Sensitivity Analysis

### 11.1 Effect of Time Horizon

Longer propagation ($N \uparrow$ or $\Delta t \uparrow$):
- ✓ More time for trajectories to separate
- ✗ Larger interval growth (more conservatism)

Shorter propagation:
- ✓ Tighter intervals
- ✗ Less time for separation

### 11.2 Effect of Initial Uncertainty

Larger $\mathcal{X}_0$:
- ✗ Larger reachable sets
- ✗ More overlap (harder to separate)

Smaller $\mathcal{X}_0$:
- ✓ Smaller reachable sets
- ✓ Easier to separate

### 11.3 Effect of Learning Rate

Larger $\alpha$:
- ✓ Faster initial convergence
- ✗ Risk of instability/oscillation

Smaller $\alpha$:
- ✓ More stable
- ✗ Slower convergence

**Typical choice:** $\alpha \in [0.01, 0.5]$

---

## 12. Mathematical Properties

### 12.1 Soundness of Interval Propagation

The interval propagation is **sound** (conservative):

$$\forall \mathbf{x}(0) \in \mathcal{X}_0, \, \mathbf{w}(t) \in \mathcal{W}, \, \mathbf{p} \in \mathcal{P}_i : \quad \mathbf{x}(T) \in \mathcal{R}_i(\mathbf{u})$$

This guarantees that all true trajectories are contained in the computed reachable set.

### 12.2 Gradient Existence

The loss function $L(\mathbf{u})$ is differentiable almost everywhere with respect to $\mathbf{u}$, except at points where:
1. Intervals transition from overlapping to non-overlapping
2. Interval boundaries align exactly

JAX's automatic differentiation handles these non-smooth points via subdifferential or generalized gradients.

### 12.3 Non-Convexity

The loss function $L(\mathbf{u})$ is generally **non-convex**:
- Multiple local minima possible
- Gradient descent may converge to suboptimal solutions
- Multi-start helps find better (near-global) solutions

---

## 13. Summary of Key Equations

| **Concept** | **Equation** |
|-------------|-------------|
| System dynamics | $\dot{\mathbf{x}} = f(t, \mathbf{x}, \mathbf{u}, \mathbf{w}, \mathbf{p})$ |
| Interval propagation | $\mathcal{X}_{k+1} = \mathcal{X}_k + \Delta t \cdot f(\mathcal{X}_k, \mathbf{u}, \mathcal{W}, \mathcal{P})$ |
| Reachable set | $\mathcal{R}_i(\mathbf{u}) = \Phi_N(\mathcal{X}_0, \mathbf{u}, \mathcal{W}, \mathcal{P}_i)$ |
| Overlap volume | $\text{overlap}(\mathcal{I}_1, \mathcal{I}_2) = \prod_{i} \max(0, \min(\overline{x}_{1,i}, \overline{x}_{2,i}) - \max(\underline{x}_{1,i}, \underline{x}_{2,i}))$ |
| Loss function | $L(\mathbf{u}) = \sum_{i<j} \text{overlap}(\mathcal{R}_i(\mathbf{u}), \mathcal{R}_j(\mathbf{u}))$ |
| Gradient descent | $\mathbf{u}_{n+1} = \mathbf{u}_n - \alpha \nabla_{\mathbf{u}} L(\mathbf{u}_n)$ |
| Optimal control | $\mathbf{u}^* = \arg\min_{\mathbf{u}} L(\mathbf{u})$ |

---

## References

This formulation is based on:
1. **Interval analysis** for reachability
2. **Natural interval extensions** for function evaluation
3. **Automatic differentiation** (JAX) for gradient computation
4. **Gradient-based optimization** for control synthesis
5. **Active fault diagnosis** for separating input design

---

**End of Mathematical Formulation**
