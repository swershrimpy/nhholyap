# Extended Quadruped Model with Hidden Angular Velocity

## Overview

This implementation extends the quadruped fault diagnosis system to include **angular velocity as a hidden state variable**, providing a more realistic model of gyroscope-based navigation.

**Key Innovation**: The yaw angle is now the **integral of angular velocity**, and sensor faults affect the gyroscope measurement, causing accumulated yaw errors over time.

---

## 🎯 System Architecture

### State Space (4D)

```
x = [px, py, yaw, omega]
```

where:
- `px, py`: Position coordinates (m)
- `yaw`: Heading angle (rad)
- **`omega`: Angular velocity (rad/s)** ← Hidden state!

### Dynamics

```
dpx/dt = vx * cos(yaw) - vy * sin(yaw)
dpy/dt = vx * sin(yaw) + vy * cos(yaw)
dyaw/dt = omega
domega/dt = alpha * u_omega
```

**Key difference from 3D model**: Yaw dynamics are now **second-order** (omega is the first derivative, making yaw the second integral of control).

---

## 🔬 Fault Models

### Actuator Fault

Affects angular acceleration:
```
domega/dt = alpha * u_omega
```
where `alpha ∈ [0.6, 0.8]` (20-40% reduction)

**Effect**: Reduces the robot's ability to change angular velocity, affecting turning performance.

### Sensor Fault (Gyroscope)

The gyroscope measures angular velocity with an **affine fault**:
```
omega_measured = a * omega_true + b
```
where:
- `a ∈ [1.25, 1.50]`: Scale factor (25-50% scaling error)
- `b ∈ [0.05, 0.15]`: Bias (rad/s)

**Accumulated Yaw Error**:

The observed yaw integrates the faulty gyro reading:
```
yaw_observed(t) = yaw(0) + ∫₀ᵗ omega_measured(s) ds
                = yaw(0) + ∫₀ᵗ (a * omega_true(s) + b) ds

yaw_error(t) = yaw_observed(t) - yaw_true(t)
             = ∫₀ᵗ ((a-1) * omega_true(s) + b) ds
```

**Key insights**:
- Scale error (a ≠ 1): Error accumulates proportional to angular velocity × time
- Bias (b ≠ 0): Error grows **linearly with time**
- Combined effect: Yaw error continuously accumulates during motion

---

## 📊 Observations

```
y = [px, py, yaw_observed]
```

**Important**: Angular velocity `omega` is a **hidden state** - not directly observed!

Observation matrix:
```python
C = [[1, 0, 0, 0],   # px
     [0, 1, 0, 0],   # py
     [0, 0, 1, 0]]   # yaw (omega not observed!)
```

---

## 🚀 Quick Start

```bash
# Run interactive demo
jupyter notebook go2_hidden_angular_velocity_demo_executed.ipynb
```

**Generated visualizations**:
- `go2_hidden_angular_velocity_trajectories.pdf` - Trajectory evolution and yaw error accumulation
- `go2_hidden_angular_velocity_final_intervals.pdf` - Final reachable sets comparison

---

## 📈 Results Summary

### Key Observations

1. **Position Trajectories**:
   - Actuator fault creates distinct divergent path
   - Sensor fault follows nominal path (no physical deviation)

2. **Yaw Evolution**:
   - Sensor fault causes growing discrepancy between true and observed yaw
   - Error accumulates continuously during motion

3. **Angular Velocity**:
   - Actuator fault reduces omega magnitude
   - Sensor fault doesn't affect true omega (only measurement)

4. **Yaw Error Accumulation**:
   - Sensor fault shows **linear growth** due to bias
   - Error magnitude depends on trajectory (more turning → more error)

### Sample Results (5-second trajectory)

| Scenario | Final Position Error | Final Yaw Error |
|----------|---------------------|-----------------|
| Nominal | 0 m | 0° |
| Sensor Fault | ~0 m (follows nominal) | ~15-40° (accumulated) |
| Actuator Fault | ~0.5-1.5 m | ~20-60° (from reduced turning) |

---

## 🔍 Comparison: 3D vs 4D Models

### Simple 3D Model

```
State: [px, py, yaw]
Dynamics: dyaw/dt = alpha * omega_cmd
Sensor fault: yaw_measured = a * yaw_true + b
Effect: Instantaneous yaw error
```

### Extended 4D Model

```
State: [px, py, yaw, omega]
Dynamics: dyaw/dt = omega, domega/dt = alpha * u_omega
Sensor fault: omega_measured = a * omega_true + b
Effect: Accumulated yaw error through integration
```

### Advantages of 4D Model

| Aspect | 3D Model | 4D Model |
|--------|----------|----------|
| **Realism** | Simplified | More realistic gyro-based navigation |
| **Yaw dynamics** | First-order | Second-order (with angular acceleration) |
| **Sensor fault** | Direct angle error | Gyroscope → integrated angle error |
| **Error behavior** | Instantaneous | Accumulates over time |
| **Hidden state** | None | Angular velocity not observed |
| **Fault distinguishability** | Limited | Better (actuator affects omega, sensor doesn't) |

---

## 🧪 Technical Implementation

### Interval Propagation

For each fault scenario, we propagate two interval sets:

1. **True state interval**: Physical robot state under actuator fault
2. **Observed state interval**: What sensors report (includes gyro fault)

```python
# Propagate true state
x_true_next = interval_dynamics_step_extended(x_true, u, alpha_int, dt)

# Apply sensor fault to get observed state
x_observed, yaw_error_int = apply_gyro_fault_to_state_interval(
    x_true, scenario, dt, yaw_error_int
)
```

### Gyroscope Fault Application

```python
# Compute gyro error rate: (a-1) * omega + b
gyro_error_rate = (a - 1) * omega_true + b

# Accumulate yaw error
yaw_error += dt * gyro_error_rate

# Observed yaw = true yaw + accumulated error
yaw_observed = yaw_true + yaw_error
```

---

## 📐 Mathematical Formulation

### System Dynamics

State evolution:
```
ẋ = f(x, u, α) = [vx*cos(θ) - vy*sin(θ)]
                  [vx*sin(θ) + vy*cos(θ)]
                  [ω                    ]
                  [α * u_ω              ]
```

where `x = [px, py, θ, ω]`, `u = [vx, vy, u_ω]`, and `α` is actuator effectiveness.

### Observation Model with Sensor Fault

True observation (no fault):
```
y = C @ x = [px, py, θ]
```

Observed with gyro fault:
```
θ_obs(t) = θ_true(t) + ∫₀ᵗ ((a-1) * ω_true(s) + b) ds
y_obs = [px, py, θ_obs]
```

---

## 🎯 Fault Diagnosis Implications

### Detectability Analysis

**Actuator Fault**:
- ✅ **Highly detectable** - affects position AND yaw
- ✅ Affects hidden state (omega) directly
- ✅ Creates distinct reachable sets in position space

**Sensor Fault**:
- ⚠️ **Not detectable from position alone** - follows nominal path
- ✅ **Detectable from yaw discrepancy** - observed yaw diverges from expected
- ✅ **Time-dependent signature** - error grows linearly with time
- ✅ Can infer from "impossible" position-yaw combinations

### Diagnosis Strategy

1. **Position-based detection**: Identifies actuator faults
2. **Yaw consistency check**: Compares observed yaw with expected yaw from position
3. **Temporal analysis**: Sensor faults show growing yaw error over time
4. **Combined diagnosis**: Actuator + sensor faults create composite signature

---

## 🔬 Experimental Insights

### Gyro Bias Accumulation

For a circular trajectory with radius R and angular velocity ω₀:

```
Total rotation: θ_total = ω₀ * T
Gyro bias error: Δθ_bias = b * T  (grows linearly!)

Example: b = 0.1 rad/s, T = 5s
→ Yaw error = 0.5 rad = 28.6°
```

### Scale Error Effects

```
Scale error accumulation: Δθ_scale = (a-1) * ∫ω dt = (a-1) * θ_total

Example: a = 1.3 (30% scale error), θ_total = π rad
→ Yaw error = 0.3π = 54°
```

---

## 🚧 Limitations and Extensions

### Current Limitations

1. **Linear dynamics**: Assumes small-angle deviations in interval propagation
2. **Constant faults**: Doesn't model time-varying faults
3. **Single gyro axis**: Only models yaw, not full 3D orientation
4. **No complementary filtering**: Real systems fuse gyro + other sensors

### Future Extensions

- [ ] Full 3D orientation with quaternions
- [ ] Magnetometer/compass sensor fusion
- [ ] Time-varying fault parameters
- [ ] Kalman filter-based yaw estimation
- [ ] Gyro drift models (random walk)
- [ ] Adaptive fault detection using residuals

---

## 📚 References

### Related Files

- **Demo**: `go2_hidden_angular_velocity_demo.ipynb`
- **Simple 3D model**: `go2_separating_input_simple.py`
- **Feedback controller**: `go2_separating_feedback_controller.py`
- **Mathematical foundation**: `MATHEMATICAL_FORMULATION.md`

### Key Concepts

- **Interval arithmetic**: Propagating uncertainty through nonlinear dynamics
- **Sensor fusion**: Combining position and orientation sensors
- **Fault diagnosis**: Distinguishing between actuator and sensor faults
- **Observability**: Hidden state (omega) inference from observations

---

## 🎉 Summary

### What This Model Adds

✅ **More realistic sensor modeling** - Gyroscope → integrated yaw
✅ **Hidden state representation** - Angular velocity not directly observed
✅ **Second-order dynamics** - Richer yaw behavior
✅ **Accumulated error modeling** - Sensor faults grow over time
✅ **Better fault distinguishability** - Actuator affects omega, sensor doesn't

### Key Takeaway

By modeling **angular velocity as a hidden state** and **yaw as its integral**, we capture the realistic behavior of gyroscope-based navigation systems where sensor faults accumulate over time through integration. This makes sensor faults more detectable through temporal analysis and yaw-position consistency checks.

---

**File**: `go2_hidden_angular_velocity_demo.ipynb`
**Date**: February 17, 2026
**System**: Extended 4D quadruped with hidden angular velocity state
