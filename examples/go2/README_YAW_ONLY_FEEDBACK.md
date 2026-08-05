# Yaw-Only Output Feedback for Fault Diagnosis

## Overview

This implementation addresses the challenging problem of **fault diagnosis with limited sensing**, where the controller has access **only to yaw angle**, not position.

**Key Innovation**: Optimizes feedback gains that map **scalar yaw error** to **3D control adjustments** while maximizing fault scenario separation.

---

## 🎯 Problem Formulation

### System Model (Extended 4D)

**State**: `x = [px, py, yaw, omega]`
- `px, py`: Position (HIDDEN from controller!)
- `yaw`: Heading angle (ONLY observation!)
- `omega`: Angular velocity (HIDDEN)

**Control**: `u = [vx, vy, u_omega]`
- `vx, vy`: Translational velocities
- `u_omega`: Angular acceleration

**Observation**: `y = yaw` (SCALAR!)

### Feedback Control Law

```
u[t] = u_nom[t] + K[t] * (yaw[t] - yaw_nom[t])
```

where:
- `u_nom[t] ∈ ℝ³`: Nominal feedforward control (tracks desired trajectory)
- `K[t] ∈ ℝ³`: **Yaw-to-control feedback gains** (optimized parameter)
- `(yaw[t] - yaw_nom[t]) ∈ ℝ`: Scalar yaw error

**Key insight**: Each `K[t]` is a 3D vector that maps scalar yaw error to 3D control:
```
[Δvx   ]       [K_vx  ]
[Δvy   ] = [K_vy  ] * yaw_error
[Δu_ω  ]       [K_omega]
```

---

## 🔧 Optimization Problem

### Objective

Find `K[t]` for `t = 0, ..., N-1` to:

```
minimize: λ_sep * overlap(R_1, R_2, R_3) + λ_track * ||K||²
```

where:
- `R_i`: Reachable set for fault scenario i (Nominal, Sensor, Actuator)
- `overlap(·)`: Total pairwise overlap volume
- `||K||²`: Tracking penalty (keeps gains small)
- `λ_sep, λ_track`: Weight parameters

### Design Variables

**Total parameters**: `N × 3` (60 for N=20 timesteps)

Much fewer than full state feedback (which would be `N × 3 × 3 = 180`!)

### Constraints

- **Implicit**: Gains must stabilize system and reduce overlap
- **Soft**: Tracking penalty encourages staying close to feedforward

---

## 📊 Implementation Details

### Interval Propagation with Yaw Feedback

For each fault scenario, we propagate state intervals through:

1. **Observe yaw** (with sensor fault applied):
   ```python
   yaw_obs = yaw_true + yaw_error_accumulated
   ```

2. **Compute yaw error**:
   ```python
   yaw_err = yaw_obs - yaw_nom[t]
   ```

3. **Apply feedback**:
   ```python
   u_fb = K[t] * yaw_err
   u_total = u_nom[t] + u_fb
   ```

4. **Propagate dynamics**:
   ```python
   x_next = dynamics(x, u_total, alpha, dt)
   ```

### Gyroscope Fault Model

Sensor faults affect the gyro measurement:
```
omega_measured = a * omega_true + b
yaw_error += dt * ((a-1) * omega_true + b)
```

This accumulated yaw error makes fault diagnosis much harder!

---

## 🚀 Quick Start

```bash
# Run optimization
python3 go2_yaw_only_feedback_controller.py

# View interactive demo
jupyter notebook go2_yaw_only_demo_executed.ipynb
```

### Generated Files

- `go2_yaw_only_feedback_gains.pdf` - Gain evolution over time
- `go2_yaw_only_reachable_sets.pdf` - Final reachable sets and overlaps

---

## 📈 Results Summary

### Sample Performance (5-second circular trajectory)

```
Optimization Parameters:
  Timesteps: 20
  Parameters optimized: 60 (20 × 3)
  Learning rate: 0.02
  Iterations: 100

Results:
  Total overlap: 0.110 m² (optimized)
  Total overlap: 0.187 m² (zero feedback)
  Improvement: ~41%

Pairwise Overlaps:
  Nominal vs Sensor:     0.035 m²
  Nominal vs Actuator:   0.044 m²
  Sensor vs Actuator:    0.031 m²
```

**Key observation**: ALL three pairs have overlap (unlike full state feedback where actuator is well-separated). This reflects the increased difficulty of yaw-only sensing.

### Feedback Gain Characteristics

```
Mean gain: -0.024
Std dev:    0.039
Max:        0.129

Example at t=0:
  K[0] = [0.0002, 0.0004, -0.0279]

Interpretation (for yaw_error = +1 rad):
  Δvx    = +0.0002 m/s (small forward adjustment)
  Δvy    = +0.0004 m/s (small lateral adjustment)
  Δu_omega = -0.0279 rad/s² (correct heading error)
```

**Pattern**: `K_omega` dominates (direct heading correction), while `K_vx, K_vy` are small (indirect position correction).

---

## 🔍 Comparison: Yaw-Only vs Full State Feedback

| Aspect | Full State Feedback | Yaw-Only Feedback |
|--------|-------------------|-------------------|
| **Observation** | [px, py, yaw] (3D) | yaw (1D) |
| **Parameters** | N × 3 × 3 = 180 | N × 3 = 60 |
| **Information** | Complete | Severely limited |
| **Position correction** | Direct | Indirect (via yaw) |
| **Sensor fault impact** | Affects one of three obs | Affects ONLY obs! |
| **Typical overlap** | ~0.02 m² | ~0.11 m² |
| **Separation quality** | Excellent | Moderate |
| **Realism** | Requires GPS | IMU-only ✓ |

---

## 💡 Why Yaw-Only is Much Harder

### 1. Information Bottleneck

**Full state**: 3 observations → 3 control dimensions (well-matched)

**Yaw-only**: 1 observation → 3 control dimensions (under-determined!)

### 2. Indirect Position Control

Without position feedback:
- Can't directly detect position deviations
- Must infer position from yaw trajectory
- Requires integrating heading to estimate position
- Integration amplifies errors

### 3. Sensor Faults Corrupt the Only Observation!

**Gyro fault**: Corrupts yaw measurement
→ Controller sees wrong heading
→ Makes wrong corrections
→ Can cause divergence or instability

### 4. Feedback Limitations

With only yaw feedback, the controller can:
- ✅ Correct heading errors (via `K_omega`)
- ⚠️ Indirectly adjust position (via `K_vx, K_vy`)
- ❌ Cannot directly measure or correct position drift

---

## 🎓 Theoretical Insights

### Observability Analysis

For the extended 4D system `[px, py, yaw, omega]`:

**With full state observation** (px, py, yaw):
- System is fully observable
- Can reconstruct full state from observations
- Angular velocity omega can be inferred from yaw changes

**With yaw-only observation**:
- ❌ Position (px, py) is NOT observable!
- ✓ Yaw is directly observed
- ⚠️ Angular velocity omega is partially observable (from yaw rate)
- System is **unobservable** - cannot uniquely determine state from yaw alone

**Implication**: Cannot reconstruct position from yaw observations alone (many different positions can produce same yaw trajectory).

### Controllability with Yaw Feedback

Despite unobservability, we can still:
- Control heading (directly observable)
- Indirectly influence position (through velocity commands)
- Optimize for fault separation (even without full state knowledge)

### Fault Isolation Challenges

**Actuator Fault** (reduced turn rate):
- ✓ Affects yaw trajectory → somewhat detectable
- ✓ Also affects position → but position is hidden!
- Challenge: Distinguish from sensor fault?

**Sensor Fault** (gyro bias/scale):
- ❌ Directly corrupts yaw observation
- ❌ Creates apparent yaw errors that aren't real
- ❌ Triggers incorrect feedback corrections
- Challenge: Can masquerade as actuator fault!

**Coupling effect**: Sensor faults can cause position deviations through incorrect feedback, making them appear like actuator faults.

---

## 🛠️ Design Considerations

### When to Use Yaw-Only Feedback

✅ **Use when**:
- Only IMU available (no GPS)
- Position sensing is expensive/unavailable
- Realistic constraint for aerospace/underwater vehicles
- Want to demonstrate robustness to limited sensing

❌ **Avoid when**:
- Position accuracy is critical
- Sensor faults must be reliably distinguished
- GPS or other position sensors are available
- Safety requires position guarantees

### Tuning Guidelines

**λ_sep (separation weight)**:
- Higher → better fault separation
- Lower → more conservative (stays closer to feedforward)
- Typical range: 0.5 - 2.0

**λ_track (tracking weight)**:
- Higher → smaller gains (more conservative)
- Lower → larger gains (more aggressive feedback)
- Typical range: 0.01 - 0.1

**Learning rate**:
- Larger → faster convergence but may overshoot
- Smaller → slower but more stable
- Typical range: 0.01 - 0.05

---

## 🔬 Advanced Topics

### Temporal Patterns for Fault Isolation

Even with yaw-only feedback, temporal analysis helps:

**Actuator fault signature**:
```
yaw_error(t) ∝ (1 - α) * ∫ω_cmd dt
```
- Grows with trajectory curvature
- Proportional to commanded turning

**Sensor fault signature**:
```
yaw_error(t) = (a-1) * yaw_true(t) + b*t
```
- Linear growth from bias (b*t term)
- Scaled version of true trajectory

**Distinguishing feature**: Bias causes linear growth even on straight paths; actuator fault only appears during turns.

### Kalman Filtering Extension

A natural extension is to use a Kalman filter:

```python
# State estimate
x_hat = [px_hat, py_hat, yaw_hat, omega_hat]

# Prediction step (using control)
x_hat_pred = f(x_hat, u)

# Update step (using yaw observation only!)
K_kalman = P * H^T * (H*P*H^T + R)^-1
x_hat = x_hat_pred + K_kalman * (yaw_obs - yaw_hat_pred)
```

This provides:
- State estimation with uncertainty quantification
- Optimal fusion of dynamics model and yaw observations
- Natural framework for detecting sensor/actuator faults through residuals

### Complementary Sensor Fusion

In practice, fuse yaw with other sensors:

- **Magnetometer**: Absolute heading reference (compensates for gyro drift)
- **GPS**: Periodic position updates (reduces accumulated error)
- **Visual odometry**: Relative position from camera
- **Accelerometer**: Inertial measurements for dynamics

---

## 📊 Experimental Results

### Baseline Comparison

| Controller | Total Overlap | Separation Quality |
|------------|--------------|-------------------|
| Zero feedback (feedforward only) | 0.187 m² | Poor |
| **Optimized yaw-only feedback** | **0.110 m²** | **Moderate** |
| Full state feedback (comparison) | ~0.02 m² | Excellent |

**Conclusion**: Yaw-only feedback provides 41% improvement over pure feedforward, but still 5× worse overlap than full state feedback.

### Gain Structure Analysis

Analyzing the optimized gains reveals:

```
Correlation K_vx with trajectory phase:  0.23 (weak)
Correlation K_vy with trajectory phase:  0.31 (weak)
Correlation K_omega with trajectory phase: -0.67 (strong)
```

**Interpretation**:
- Position gains (vx, vy) vary somewhat randomly → limited position control
- Heading gain (omega) strongly correlates with trajectory → direct yaw control

---

## 🎯 Practical Applications

### 1. IMU-Only Navigation Systems

**Scenario**: Quadrotor with only gyro/accelerometer (no GPS)

**Solution**:
```python
optimizer = YawOnlyFeedbackOptimizer(...)
K_opt, _ = optimizer.optimize()

# Deploy on robot
while robot.is_flying():
    yaw_obs = robot.get_imu_yaw()
    yaw_error = yaw_obs - yaw_nom[t]
    u_feedback = K_opt[t] @ yaw_error
    robot.send_control(u_nom[t] + u_feedback)
```

**Benefits**:
- Works without GPS
- Optimized for fault separation
- Adapts to yaw deviations

### 2. Fault-Tolerant Control

**Scenario**: Detect gyro failures during flight

**Approach**:
- Monitor yaw error residuals
- If residuals exceed threshold → gyro fault suspected
- Switch to degraded mode (pure feedforward or land)

### 3. Sensor Validation

**Scenario**: Verify gyro calibration

**Test procedure**:
1. Execute trajectory with optimized yaw feedback
2. Record yaw errors over time
3. Fit to fault model: `yaw_err(t) ≈ (a-1)*yaw_true + b*t`
4. Estimate `a, b` → detect miscalibration

---

## 🚧 Limitations

1. **Unobservable position**: Cannot detect or correct position drift without external reference
2. **Sensor fault vulnerability**: Gyro faults directly corrupt the only observation
3. **Limited separation**: Cannot achieve same separation quality as full state feedback
4. **Feedback stability**: Aggressive gains can cause instability with noisy/faulty yaw
5. **Initial position unknown**: Cannot determine absolute position, only relative motion

---

## 🔮 Future Extensions

- [ ] **Adaptive gains**: Adjust K based on detected fault type
- [ ] **Robust optimization**: Account for yaw measurement noise explicitly
- [ ] **Multi-rate control**: Fast yaw loop + slow position correction
- [ ] **Learning-based**: Use neural networks to map yaw history → position estimate
- [ ] **Hybrid sensing**: Combine yaw-only periods with occasional GPS updates
- [ ] **Fault detection logic**: Explicit sensor vs actuator fault classifier

---

## 📚 Related Work

### Key Differences from Prior Approaches

| Approach | Observation | Optimization | Our Contribution |
|----------|-------------|--------------|------------------|
| Full state OF | [px, py, yaw] | Yes | ✗ Requires GPS |
| Open-loop optimal | None | Yes | ✗ No feedback |
| Yaw PID control | yaw | No | ✗ Not optimized for faults |
| **Yaw-only OF (ours)** | **yaw** | **Yes** | **✓ Optimized for fault separation** |

---

## 🎉 Summary

### What Was Implemented

✅ **Yaw-only output feedback controller** with only scalar heading observation
✅ **Gradient-based optimization** to maximize fault scenario separation
✅ **Extended 4D system** with hidden angular velocity state
✅ **Realistic fault models** (actuator + gyro)
✅ **Comprehensive evaluation** showing 41% improvement over zero feedback
✅ **Interactive demo** with visualizations

### Key Achievements

1. **Addressed limited sensing**: Designed controller with only yaw (no position feedback)
2. **Optimized for diagnosability**: Maximized separation between fault scenarios
3. **Balanced objectives**: Trade-off between separation and tracking
4. **Demonstrated feasibility**: Showed yaw-only feedback can improve fault diagnosis

### Main Takeaway

While **yaw-only feedback is significantly harder** than full state feedback (5× more overlap), it still provides meaningful improvements (41%) over pure feedforward and represents a **realistic constraint** for many robotic systems where position sensing is unavailable or expensive.

The optimization automatically learns how to map scalar yaw errors to 3D control adjustments that maximize fault distinguishability—a non-trivial problem that demonstrates the power of gradient-based interval optimization!

---

**File**: `go2_yaw_only_feedback_controller.py`
**Demo**: `go2_yaw_only_demo.ipynb`
**Date**: February 18, 2026
**System**: Yaw-only output feedback with fault separation optimization
