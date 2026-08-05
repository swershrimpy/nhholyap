# Fault Diagnosis System - File Index

## 🚀 Quick Start

**New to this project?** Choose your approach:

### Passive Fault Diagnosis (Estimation)
1. Read `README_FAULT_DIAGNOSIS.md` (quick overview)
2. Run `python3 example_fault_diagnosis.py` (simple demo)
3. Open `fault_diagnosis_demo_executed.ipynb` in Jupyter (interactive demo)

### Active Fault Diagnosis (Separating Inputs)
1. Read `README_SEPARATING_INPUT.md` (overview)
2. Run `python3 go2_separating_input_simple.py` (optimization demo)
3. Open `go2_separating_input_demo_executed.ipynb` in Jupyter (visualizations)

---

## 📁 File Organization

### 🔧 Core Implementation

**Passive Approach** (estimate faults from observations):
- `fault_diagnosis.py` - Nonlinear optimization-based fault estimation
- `custom_controller.py` - Robot controller with fault diagnosis integration

**Active Approach** (optimize inputs for fault separation):
- `go2_separating_input_simple.py` - Separating input optimizer (self-contained)
- `go2_separating_input.py` - Alternative implementation using immrax

### 📖 Documentation
- `README_FAULT_DIAGNOSIS.md` - **Passive approach** - Quick start guide
- `README_SEPARATING_INPUT.md` - **Active approach** - Separating inputs guide
- `MATHEMATICAL_FORMULATION.md` - Complete mathematical details
- `FAULT_DIAGNOSIS.md` - Technical documentation (passive)
- `DEMO_README.md` - Interactive demo guide
- `IMPLEMENTATION_SUMMARY.md` - Verification and validation results
- `COMPLETE_DELIVERY.md` - Full delivery summary

### 🎯 Examples

**Passive Fault Diagnosis**:
- `example_fault_diagnosis.py` - Simple example
- `test_fault_diagnosis.py` - Comprehensive test suite
- `fault_diagnosis_demo.ipynb` - Interactive notebook demo
- `run_notebook_demo.py` - Script to execute demo notebook

**Active Fault Diagnosis**:
- `go2_separating_input_simple.py` - **Optimization example** - Run this!
- `go2_separating_input_demo.ipynb` - **Interactive visualization**

### 📊 Generated Outputs

**Passive Approach**:
- `demo1_actuator_fault.pdf` - Actuator fault visualization
- `demo2_sensor_fault.pdf` - Sensor fault visualization
- `demo3_combined_faults.pdf` - Combined faults visualization
- `demo_summary.pdf` - Comparison summary
- `fault_diagnosis_demo_executed.ipynb` - Executed demo with outputs

**Active Approach**:
- `go2_separating_input_reachable_sets.pdf` - Reachable set visualization
- `go2_input_comparison.pdf` - Input strategy comparison
- `go2_separating_input_demo_executed.ipynb` - Executed optimization demo

---

## 🎓 Learning Path

### Beginner
1. `README_FAULT_DIAGNOSIS.md` - Understand the system
2. `example_fault_diagnosis.py` - Run simple example
3. `DEMO_README.md` - Learn about the demo

### Intermediate
1. `fault_diagnosis_demo.ipynb` - Interactive exploration
2. `test_fault_diagnosis.py` - See comprehensive tests
3. `FAULT_DIAGNOSIS.md` - Technical deep dive

### Advanced
1. `MATHEMATICAL_FORMULATION.md` - Mathematical theory
2. `fault_diagnosis.py` - Study implementation
3. `IMPLEMENTATION_SUMMARY.md` - Review validation

---

## 🔍 Find What You Need

### "I want to understand the math"
→ `MATHEMATICAL_FORMULATION.md`

### "I want to run a quick demo"
**Passive**: `python3 example_fault_diagnosis.py`
**Active**: `python3 go2_separating_input_simple.py`

### "I want interactive visualizations"
**Passive**: `jupyter notebook fault_diagnosis_demo_executed.ipynb`
**Active**: `jupyter notebook go2_separating_input_demo_executed.ipynb`

### "I want to optimize control for fault detection"
→ `README_SEPARATING_INPUT.md` + `go2_separating_input_simple.py`

### "I want to estimate fault parameters"
→ `README_FAULT_DIAGNOSIS.md` + `fault_diagnosis.py`

### "I want to see test results"
→ `python3 test_fault_diagnosis.py`

### "I want API documentation"
→ `README_FAULT_DIAGNOSIS.md` + `FAULT_DIAGNOSIS.md`

### "I want to integrate with robot"
→ `custom_controller.py` + `README_FAULT_DIAGNOSIS.md`

### "I want to verify the implementation"
→ `IMPLEMENTATION_SUMMARY.md`

---

## 📋 File Descriptions

### Core Implementation
| File | Purpose | Size |
|------|---------|------|
| **fault_diagnosis.py** | Passive fault estimation | 17 KB |
| **go2_separating_input_simple.py** | Active separating input optimizer | 14 KB |
| **go2_separating_input.py** | Alternative (immrax-based) | 18 KB |
| **custom_controller.py** | Robot controller integration | 22 KB |

### Documentation
| File | Purpose | Size |
|------|---------|------|
| **README_FAULT_DIAGNOSIS.md** | Passive approach quick start | 8.7 KB |
| **README_SEPARATING_INPUT.md** | Active approach guide | 15 KB |
| **MATHEMATICAL_FORMULATION.md** | Complete math derivation | 13 KB |
| **FAULT_DIAGNOSIS.md** | Technical documentation | 11 KB |
| **DEMO_README.md** | Demo instructions | 8 KB |
| **IMPLEMENTATION_SUMMARY.md** | Validation results | 6 KB |
| **COMPLETE_DELIVERY.md** | Delivery summary | 10 KB |

### Examples & Tests
| File | Purpose | Size |
|------|---------|------|
| **example_fault_diagnosis.py** | Simple passive example | 7.1 KB |
| **test_fault_diagnosis.py** | Comprehensive test suite | 16 KB |
| **fault_diagnosis_demo.ipynb** | Passive demo notebook | 39 KB |
| **go2_separating_input_demo.ipynb** | Active demo notebook | 19 KB |
| **run_notebook_demo.py** | Script to execute notebooks | 1.8 KB |

---

## ✅ Recommended Reading Order

1. `README_FAULT_DIAGNOSIS.md` (5 min) - Overview
2. Run `example_fault_diagnosis.py` (1 min) - See it work
3. `DEMO_README.md` (3 min) - Demo guide
4. Run `fault_diagnosis_demo.ipynb` (10 min) - Interactive exploration
5. `FAULT_DIAGNOSIS.md` (15 min) - Technical details
6. `MATHEMATICAL_FORMULATION.md` (20 min) - Mathematical foundation

**Total time**: ~1 hour to understand the complete system

---

## 🎯 By Use Case

### Research/Academic
- `MATHEMATICAL_FORMULATION.md` - Theory
- `IMPLEMENTATION_SUMMARY.md` - Validation
- `fault_diagnosis.py` - Implementation

### Industry/Engineering  
- `README_FAULT_DIAGNOSIS.md` - Quick start
- `example_fault_diagnosis.py` - Usage pattern
- `custom_controller.py` - Integration

### Education/Learning
- `fault_diagnosis_demo.ipynb` - Interactive learning
- `DEMO_README.md` - Guided tutorial
- `test_fault_diagnosis.py` - Examples

---

## 🔬 Two Complementary Approaches

### Passive Fault Diagnosis
**What**: Estimate fault parameters from observations after execution
**When**: You have trajectory data and want fault magnitude/type
**Output**: `[px, py, yaw]` error + fault parameters (α, a, b)
**Files**: `fault_diagnosis.py`, `example_fault_diagnosis.py`

### Active Fault Diagnosis
**What**: Optimize control inputs to maximize fault distinguishability
**When**: You want to design diagnostic trajectories
**Output**: Optimal input `[vx, vy, ω]` + reachable set separation
**Files**: `go2_separating_input_simple.py`, `go2_separating_input_demo.ipynb`

**Use Together**: Run active approach to design optimal input, then use passive approach to estimate exact fault parameters!

---

**Last Updated**: February 17, 2026
**Total Files**: 28+
**Total Documentation**: ~4,000 lines
**Total Code**: ~4,000 lines
**Approaches**: 2 (Passive + Active)
