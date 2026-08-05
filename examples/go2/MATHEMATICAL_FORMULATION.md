# Mathematical Formulation: Vision-Based Output Feedback Control for Unitree Go2

## Overview

This document describes the mathematical formulation of a vision-based output feedback control system for the Unitree Go2 quadruped robot. The system uses learned visual features (DINOv2) combined with trajectory optimization and sensitivity-based feedback control to achieve robust path tracking under uncertain observations.

---

## Problem Setup

### System Dynamics

The Go2 robot is modeled as a unicycle-like system with state-space representation:

**State Vector** $\mathbf{x} \in \mathbb{R}^{n_x}$:
```
x = [x, y, θ]ᵀ
```
where:
- $x, y$: Position in the 2D plane (meters)
- $\theta$: Heading angle (radians)

**Control Input** $\mathbf{u} \in \mathbb{R}^{n_u}$:
```
u = [vₓ, vᵧ, ωᵧ]ᵀ
```
where:
- $v_x$: Forward velocity (m/s)
- $v_y$: Lateral velocity (m/s)
- $\omega_y$: Yaw rate (rad/s)

**Dimensions**:
- $n_x = 3$ (state dimension)
- $n_u = 3$ (control input dimension)
- $n_y = 3$ (observation dimension)

### Discrete-Time Dynamics

The system evolves according to:

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$$

where $\mathbf{w}_t$ represents process disturbances.

**Nominal Dynamics** (without disturbances):
$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$$

**Time discretization**: $\Delta t = 0.5$ seconds

---

## Observation Model

### Vision-Based Observations

The system uses camera images processed through a learned observation model:

#### 1. **Feature Extraction (DINOv2)**

Raw camera images $\mathbf{I}_t \in \mathbb{R}^{H \times W \times 3}$ are processed through a pre-trained DINOv2 vision transformer:

$$\mathbf{d}_t = \text{DINOv2}(\mathbf{I}_t) \in \mathbb{R}^{1536}$$

**DINOv2 Configuration**:
- Model: ViT-g/14 (giant vision transformer with 14×14 patch size)
- Output: CLS token features (global image descriptor)
- Dimension: 1536-dimensional feature vector
- Pre-processing: Resize to 224×224, normalize with ImageNet statistics

#### 2. **Learned Observation Function**

The DINOv2 features are mapped to a low-dimensional observation space through a learned neural network:

$$\mathbf{y}_t = h_{\text{NN}}(\mathbf{d}_t) \in \mathbb{R}^{n_y}$$

**Network Architecture** (`SupervisedDinoObservability`):
```
Input: 1536-dim (DINOv2 features)
  ↓
Linear(1536 → 512) + GELU
  ↓
Linear(512 → 512) + GELU
  ↓
Linear(512 → 512) + GELU
  ↓
Linear(512 → 3)
  ↓
Output: 3-dim observations
```

#### 3. **Linear Observation Model**

The observations are related to the robot state through a learned linear observation matrix:

$$\mathbf{y}_t \approx \mathbf{C}(\mathbf{x}_t + \mathbf{x}_{\text{offset}}) + \boldsymbol{\eta}_t$$

where:
- $\mathbf{C} \in \mathbb{R}^{n_y \times n_x}$: Learned observation matrix
- $\mathbf{x}_{\text{offset}} = [-0.46, 0.34, 0]^T$: State offset for coordinate transformation
- $\boldsymbol{\eta}_t$: Observation noise

**Observation Matrix** (from trained model):
```
C = [[-0.0329,  0.9805, -0.1938],
     [-0.8052, -0.5551, -0.2087],
     [-0.8518, -0.4816,  0.2061]]
```

---

## Trajectory Optimization Problem

### Nominal Trajectory Planning

A nominal open-loop trajectory is pre-computed offline:

**Optimization Variables**:
- State trajectory: $\{\bar{\mathbf{x}}_0, \bar{\mathbf{x}}_1, \ldots, \bar{\mathbf{x}}_N\}$
- Control trajectory: $\{\bar{\mathbf{u}}_0, \bar{\mathbf{u}}_1, \ldots, \bar{\mathbf{u}}_{N-1}\}$

**Constraints**:
1. Initial condition: $\bar{\mathbf{x}}_0 = \mathbf{x}_{\text{init}}$
2. Dynamics: $\bar{\mathbf{x}}_{t+1} = f(\bar{\mathbf{x}}_t, \bar{\mathbf{u}}_t)$ for $t = 0, \ldots, N-1$
3. Goal condition: $\|\bar{\mathbf{x}}_N - \mathbf{x}_{\text{goal}}\| \leq \epsilon$

**Stored in**: `Go2_OF_Perception.npz` or `Go2_OF_Perception2.npz`

---

## Output Feedback Control

### Sensitivity-Based Feedback Controller

The controller uses **first-order sensitivity matrices** to correct for observation deviations from the nominal trajectory.

#### Sensitivity Matrices

Four sensitivity matrices are pre-computed via trajectory optimization:

1. **$\boldsymbol{\Phi}_{xx} \in \mathbb{R}^{Nn_x \times Nn_x}$**: State-to-state sensitivity
   - Captures how perturbations in initial state affect future states

2. **$\boldsymbol{\Phi}_{ux} \in \mathbb{R}^{Nn_u \times Nn_x}$**: Control-to-state sensitivity
   - Captures how control corrections affect future states

3. **$\boldsymbol{\Phi}_{xy} \in \mathbb{R}^{Nn_x \times Nn_y}$**: State-to-observation sensitivity
   - Captures how state perturbations affect observations

4. **$\boldsymbol{\Phi}_{uy} \in \mathbb{R}^{Nn_u \times Nn_y}$**: Control-to-observation sensitivity
   - Captures how control corrections affect observations

#### Feedback Gain Matrix

The output feedback gain matrix is computed as:

$$\mathbf{K} = \boldsymbol{\Phi}_{uy} - \boldsymbol{\Phi}_{ux} \boldsymbol{\Phi}_{xx}^{-1} \boldsymbol{\Phi}_{xy}$$

where $\mathbf{K} \in \mathbb{R}^{Nn_u \times Nn_y}$ is structured as a block matrix:

$$\mathbf{K} = \begin{bmatrix}
\mathbf{K}_{0,0} & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{K}_{1,0} & \mathbf{K}_{1,1} & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{K}_{N-1,0} & \mathbf{K}_{N-1,1} & \cdots & \mathbf{K}_{N-1,N-1}
\end{bmatrix}$$

where each $\mathbf{K}_{i,j} \in \mathbb{R}^{n_u \times n_y}$.

#### Control Law

At time step $t \in \{1, 2, \ldots, N\}$, the control input is computed as:

$$\mathbf{u}_t = \bar{\mathbf{u}}_t + \Delta \mathbf{u}_t$$

where the correction term is:

$$\Delta \mathbf{u}_t = \sum_{j=0}^{t} \mathbf{K}_{t,j} \left(\mathbf{y}_j - \mathbf{C}(\bar{\mathbf{x}}_j + \mathbf{x}_{\text{offset}})\right)$$

**Expanded form**:
```
Δuₜ = K[t,0] · (y₀ - C·(x̄₀ + x_offset))
    + K[t,1] · (y₁ - C·(x̄₀ + x_offset))     # Note: uses x̄₀, not x̄₁
    + K[t,2] · (y₂ - C·(x̄₁ + x_offset))
    + ...
    + K[t,t] · (yₜ - C·(x̄ₜ₋₁ + x_offset))
```

**Note**: There is a one-step lag in the state indexing within the observation prediction, where observation $\mathbf{y}_j$ is compared against $\mathbf{C}(\bar{\mathbf{x}}_{j-1} + \mathbf{x}_{\text{offset}})$ for $j > 0$.

---

## Robustness: Tubes and Backoff

### Reachable Sets

The system computes forward reachable tubes to bound trajectory uncertainty:

- **`tube`** (intermediate backoff): Bounds on states during trajectory execution
- **`tube_f`** (final backoff): Bounds on final state at goal

These tubes account for:
1. Observation uncertainty $\boldsymbol{\eta}_t$
2. Process noise $\mathbf{w}_t$
3. Model mismatch between learned observation function and true system

---

## Problem Inputs

### Required Data Files

1. **`Go2_OF_Perception.npz`** or **`Go2_OF_Perception2.npz`**:
   - `nominal_input` ($\bar{\mathbf{u}}$): Nominal control sequence $\in \mathbb{R}^{n_u \times N}$
   - `nominal_traj` ($\bar{\mathbf{x}}$): Nominal state sequence $\in \mathbb{R}^{n_x \times (N+1)}$
   - `Phi_xx`: State-to-state sensitivity $\in \mathbb{R}^{Nn_x \times Nn_x}$
   - `Phi_ux`: Control-to-state sensitivity $\in \mathbb{R}^{Nn_u \times Nn_x}$
   - `Phi_xy`: State-to-observation sensitivity $\in \mathbb{R}^{Nn_x \times Nn_y}$
   - `Phi_uy`: Control-to-observation sensitivity $\in \mathbb{R}^{Nn_u \times Nn_y}$
   - `backoff` (tube): Intermediate reachable set bounds
   - `backoff_f` (tube_f): Final reachable set bounds

2. **`model_go2_val.pt`**: Trained neural network weights for `SupervisedDinoObservability`

3. **`learn_model.yaml`**: Network configuration parameters

### Initial State

The robot initializes its state estimate through odometry:

**Raw odometry** from robot:
- Position: $[x_{\text{raw}}, y_{\text{raw}}, z_{\text{raw}}]$ (from `SportModeState`)
- Orientation: Roll-pitch-yaw angles $[\phi, \theta, \psi]$

**Coordinate transformation**:

At initialization ($t=0$):
1. Store initial pose: $\mathbf{x}_{\text{init}} = [x_{\text{raw},0}, y_{\text{raw},0}, \psi_0]$
2. Compute rotation matrix:
   $$\mathbf{M}_{\text{rot}} = \begin{bmatrix}
   \cos(\psi_0) & \sin(\psi_0) \\
   -\sin(\psi_0) & \cos(\psi_0)
   \end{bmatrix}$$

At time $t > 0$:
1. Transform position:
   $$\begin{bmatrix} x_t \\ y_t \end{bmatrix} = \mathbf{M}_{\text{rot}} \begin{bmatrix} x_{\text{raw},t} \\ y_{\text{raw},t} \end{bmatrix} - \begin{bmatrix} x_{\text{init}} \\ y_{\text{init}} \end{bmatrix}$$

2. Wrap heading angle:
   $$\theta_t = \text{wrap}(\psi_t - \psi_0) \in [-2\pi, 0]$$

---

## Problem Outputs

### Control Commands

At each time step $t$, the controller outputs:

$$\mathbf{u}_t = [v_{x,t}, v_{y,t}, \omega_{y,t}]^T$$

These commands are sent to the robot via:
```python
sport_client.Move(u[0], u[1], u[2])
```

**Note**: In the simplified `start()` method, only $v_x$ and $\omega_y$ are used (lateral velocity $v_y = 0$).

### State Trajectory

The actual robot state trajectory is recorded:

$$\{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_N\}$$

where each $\mathbf{x}_t$ is obtained from the transformed odometry.

### Observations

Vision-based observations at each time step:

$$\{\mathbf{y}_0, \mathbf{y}_1, \ldots, \mathbf{y}_N\}$$

where:
- $\mathbf{y}_t = h_{\text{NN}}(\text{DINOv2}(\mathbf{I}_t)) \in \mathbb{R}^3$

---

## Algorithm Summary

### Visuomotor Control Loop

**Initialization**:
1. Load nominal plan: $\{\bar{\mathbf{x}}_t, \bar{\mathbf{u}}_t\}_{t=0}^N$ and sensitivity matrices
2. Compute feedback gain matrix: $\mathbf{K} = \boldsymbol{\Phi}_{uy} - \boldsymbol{\Phi}_{ux} \boldsymbol{\Phi}_{xx}^{-1} \boldsymbol{\Phi}_{xy}$
3. Initialize state estimate and observation history

**For** $t = 1, 2, \ldots, N$:

1. **Capture image**: $\mathbf{I}_t$ from robot camera

2. **Extract features**:
   - $\mathbf{d}_t = \text{DINOv2}(\mathbf{I}_t)$
   - $\mathbf{y}_t = h_{\text{NN}}(\mathbf{d}_t)$

3. **Compute control correction**:
   $$\Delta \mathbf{u}_t = \sum_{j=0}^{t} \mathbf{K}_{t,j} \left(\mathbf{y}_j - \mathbf{C}(\bar{\mathbf{x}}_j + \mathbf{x}_{\text{offset}})\right)$$

4. **Apply control**:
   $$\mathbf{u}_t = \bar{\mathbf{u}}_t + \Delta \mathbf{u}_t$$

5. **Send command**: `sport_client.Move(u[0], u[1], u[2])`

6. **Wait**: $\Delta t = 0.5$ seconds

7. **Update state estimate**: Read odometry and transform coordinates

**End For**

---

## Key Design Choices

### 1. **Why DINOv2?**
- Self-supervised learning provides robust visual features
- No need for task-specific image labels
- Strong generalization across lighting and viewpoint changes
- High-dimensional features (1536-dim) capture rich visual information

### 2. **Why Learned Observation Model?**
- Direct mapping from vision to low-dimensional observations
- Trained to approximate the observability structure $\mathbf{y} \approx \mathbf{C}\mathbf{x}$
- Reduces dimensionality: 1536 → 3
- Enables gradient-based trajectory optimization

### 3. **Why Sensitivity-Based Control?**
- First-order approximation enables real-time feedback
- Pre-computed gains avoid online optimization
- Causal structure (lower-triangular $\mathbf{K}$) ensures implementability
- Accounts for observation history through receding-horizon structure

### 4. **Why Output Feedback?**
- Directly uses observations without state estimation
- Robust to model mismatch in observation function
- No need for Extended Kalman Filter or particle filter
- Uncertainty propagation handled through sensitivity analysis

---

## Limitations and Extensions

### Current Limitations:
1. **Open-loop plan**: Nominal trajectory must be feasible
2. **First-order approximation**: Large deviations may violate linearity assumptions
3. **Pre-computed gains**: Cannot adapt to environment changes online
4. **Fixed horizon**: Trajectory length $N$ is predetermined

### Possible Extensions:
1. **Model Predictive Control (MPC)**: Replan online based on observations
2. **Adaptive gains**: Update $\mathbf{K}$ based on observed performance
3. **Obstacle avoidance**: Incorporate dynamic constraints
4. **Multi-modal observations**: Fuse camera with IMU and LiDAR
5. **Learned dynamics**: Replace hand-designed $f(\cdot)$ with neural network

---

## References

**DINOv2**:
- Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision"
- Repository: https://github.com/facebookresearch/dinov2

**Output Feedback Control**:
- Based on sensitivity-based trajectory optimization with learned observation models

**Unitree SDK**:
- Repository: https://github.com/unitreerobotics/unitree_sdk2_python
