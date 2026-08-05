# Fault Diagnosis Demo - Quick Start Guide

## 🎯 What This Demo Shows

This interactive Jupyter notebook demonstrates the complete fault diagnosis system for the Unitree Go2 robot with:

✅ **3 Comprehensive Demos**:
1. Actuator fault only (40% turn rate reduction)
2. Sensor fault only (IMU drift)
3. Combined faults (both actuator and sensor)

✅ **Visual Outputs**:
- Trajectory comparisons (nominal vs faulty)
- Heading evolution plots
- Error metrics and fault parameter estimation
- Summary comparisons across all scenarios

✅ **Complete Results**: Shows the `[px, py, yaw]` output format

---

## 🚀 Quick Start

### Method 1: Run Pre-executed Notebook (Fastest)

```bash
jupyter notebook fault_diagnosis_demo_executed.ipynb
```

This opens a notebook with all outputs already generated.

### Method 2: Execute from Scratch

```bash
# Option A: Run the script
python3 run_notebook_demo.py

# Option B: Open in Jupyter and run all cells
jupyter notebook fault_diagnosis_demo.ipynb
```

---

## 📊 Generated Outputs

After running the demo, you'll get:

### Visualizations (PDFs)

1. **`demo1_actuator_fault.pdf`** (32 KB)
   - Trajectory deviation from actuator fault
   - Heading evolution comparison
   - Error metrics breakdown

2. **`demo2_sensor_fault.pdf`** (32 KB)
   - True vs measured heading (with IMU drift)
   - Measurement error over time
   - Sensor fault parameter comparison

3. **`demo3_combined_faults.pdf`** (41 KB)
   - Combined fault scenario visualization
   - Fault parameter estimation accuracy
   - Error evolution over trajectory

4. **`demo_summary.pdf`** (28 KB)
   - Side-by-side comparison of all 3 demos
   - Position and heading error comparison
   - Fault severity assessment

### Executed Notebook

- **`fault_diagnosis_demo_executed.ipynb`** (390 KB)
  - Interactive notebook with all outputs
  - Detailed explanations and code
  - Can be re-run or modified

---

## 📋 Demo Structure

### Demo 1: Actuator Fault
```
Input:  α = 0.6 (40% turn rate reduction)
Output: [px=-3.86m, py=+3.66m, yaw=-1.80rad]
Result: ✓ 99.95% accuracy in fault estimation
```

### Demo 2: Sensor Fault
```
Input:  a=1.08, b=0.175rad (8% scale, 10° bias)
Output: [px=+0.00m, py=+0.00m, yaw=+0.00rad]
Result: ✓ 96% accuracy in scale/bias estimation
```

### Demo 3: Combined Faults
```
Input:  α=0.7, a=1.05, b=0.12rad
Output: [px=-2.55m, py=+8.06m, yaw=-1.23rad]
Result: ✓ 99% accuracy in combined fault estimation
```

---

## 🔧 Requirements

The demo requires these Python packages:

```bash
# Core dependencies (already installed)
numpy scipy matplotlib seaborn pandas

# For notebook execution
jupyter nbconvert

# Fault diagnosis module (included)
fault_diagnosis.py
```

All dependencies were installed during the initial setup.

---

## 📖 Notebook Contents

### Section 1: Setup
- Create nominal circular trajectory
- Initialize fault diagnosis system
- Configure observation model

### Section 2-4: Three Demos
- Demo 1: Actuator fault detection
- Demo 2: Sensor fault detection
- Demo 3: Combined fault detection

### Section 5: Summary Table
- Comprehensive comparison
- Performance metrics
- Visual summary

### Section 6: Key Takeaways
- Algorithm performance analysis
- Output format explanation
- Practical applications

### Section 7: Next Steps
- How to use with real robot
- Integration instructions

---

## 🎨 Example Visualizations

### Trajectory Comparison
Shows nominal (blue) vs faulty (red) trajectories with error arrows indicating the `[px, py]` components.

### Heading Evolution
Time-series plots showing how heading errors accumulate due to actuator or sensor faults.

### Error Metrics
Bar charts and comparison plots showing:
- Position errors (px, py)
- Heading errors (yaw)
- Fault parameter estimation accuracy

### Summary Dashboard
Comprehensive comparison across all three demo scenarios.

---

## 💡 Customization

You can modify the notebook to test different scenarios:

### Change Fault Severity

```python
# More severe actuator fault
actuator_fault = FaultParameters(
    actuator_effectiveness=0.4,  # 60% reduction
    sensor_scale=1.0,
    sensor_bias=0.0
)

# Larger sensor drift
sensor_fault = FaultParameters(
    actuator_effectiveness=1.0,
    sensor_scale=1.15,  # 15% scale error
    sensor_bias=0.3     # ~17° bias
)
```

### Different Trajectory Types

```python
# Figure-8 trajectory
def create_figure8_trajectory(N=30, dt=0.5):
    # ... custom trajectory code ...
    pass
```

### Adjust Noise Levels

```python
# Add more/less observation noise
measured_obs += 0.05 * np.random.randn(*measured_obs.shape)  # Higher noise
```

---

## 🔍 Understanding the Output

### Output Format: `[px, py, yaw]`

- **px** (meters): X-direction position error
  - Positive: Robot went too far forward
  - Negative: Robot didn't go far enough

- **py** (meters): Y-direction position error
  - Positive: Robot drifted to the right
  - Negative: Robot drifted to the left

- **yaw** (radians): Heading angle error
  - Positive: Robot turned more than expected
  - Negative: Robot didn't turn enough

### Example Interpretation

```python
Output: [px=-3.86, py=+3.66, yaw=-1.80]
```

Interpretation:
- Robot fell short by 3.86m in forward direction
- Robot drifted right by 3.66m
- Robot underturned by 103° (1.80 rad)
- **Total position error**: 5.32m from goal

---

## 📊 Performance Metrics

From the demo results:

| Metric | Value |
|--------|-------|
| **Actuator fault accuracy** | 99.95% |
| **Sensor fault accuracy** | 96-98% |
| **Position error accuracy** | >99% |
| **Heading error accuracy** | >99% |
| **Computation time** | 1-5 seconds |

---

## 🚨 Troubleshooting

### Issue: Notebook won't execute

**Solution**:
```bash
# Reinstall dependencies
pip install jupyter nbconvert matplotlib seaborn pandas

# Try the standalone script
python3 run_notebook_demo.py
```

### Issue: Plots not displaying

**Solution**:
- Ensure matplotlib backend is configured
- Try running in Jupyter Lab instead of Jupyter Notebook
- Check that PDFs were generated (open them separately)

### Issue: Import errors

**Solution**:
```bash
# Verify fault_diagnosis module is accessible
python3 -c "from fault_diagnosis import FaultDiagnosisGo2"

# If error, ensure you're in the correct directory
cd /path/to/go2/
```

---

## 📚 Additional Resources

- **Mathematical formulation**: See `MATHEMATICAL_FORMULATION.md`
- **Detailed documentation**: See `FAULT_DIAGNOSIS.md`
- **API reference**: See `README_FAULT_DIAGNOSIS.md`
- **Test suite**: Run `python3 test_fault_diagnosis.py`
- **Simple example**: Run `python3 example_fault_diagnosis.py`

---

## ✅ Verification Checklist

After running the demo, verify:

- [ ] All 4 PDF visualizations generated
- [ ] Executed notebook saved
- [ ] Demo 1 shows ~4m position error for 40% actuator fault
- [ ] Demo 2 shows minimal position error for sensor-only fault
- [ ] Demo 3 shows combined effects of both faults
- [ ] Summary table displays all results
- [ ] All fault estimates within 5% of true values

---

## 🎓 Learning Objectives

By the end of this demo, you should understand:

1. ✅ How actuator faults affect robot trajectories
2. ✅ How sensor faults (IMU drift) impact localization
3. ✅ How the diagnosis algorithm separates fault types
4. ✅ What the `[px, py, yaw]` output represents
5. ✅ How to interpret fault severity
6. ✅ When to recommend maintenance or recalibration

---

## 🎯 Next Steps

### For Simulation/Testing
Continue with `test_fault_diagnosis.py` for more scenarios

### For Real Robot Deployment
1. Review `custom_controller.py` integration
2. Test with `start_with_fault_diagnosis()` method
3. Deploy online monitoring with `online_fault_monitoring()`

### For Research/Development
1. Modify fault models in `fault_diagnosis.py`
2. Add new fault types
3. Extend to other robot platforms

---

## 📞 Support

For questions or issues:
- Check documentation files (*.md)
- Review test suite for examples
- Open an issue in the repository

---

**Demo Version**: 1.0
**Last Updated**: February 2026
**Compatible with**: Python 3.10+, Jupyter Notebook 6.0+
