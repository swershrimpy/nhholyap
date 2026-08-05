# Animated 3-D Refinement Interval Plot

This directory contains tools for visualising the interval refinement process for the faulty nonholonomic car example.

## Files

| File | Description |
|------|-------------|
| `plot_refinement_3d.py` | Static 3-D plot showing all four panels (unrefined + three refined pairs) |
| `plot_refinement_3d_with_traj.py` | Static plot with a single trajectory overlay (user-specified fault mode) |
| `animate_refinement_3d.py` | Animated video showing interval evolution, trajectory, and dynamic validity |
| `ANIMATION_README.md` | This file |

---

## Design Requirements

### 1. Static Plot (`plot_refinement_3d.py`)

- **3-D view**: `px` (x-axis), `time` (y-axis), `py` (z-axis)
- **Four panels**:
  - Panel A: Unrefined output intervals for all three scenarios
  - Panel B: Refined intervals for Nominal vs Actuator Fault pair
  - Panel C: Refined intervals for Nominal vs Sensor Fault pair
  - Panel D: Refined intervals for Actuator vs Sensor Fault pair
- **Interval boxes**: Coloured rectangles at each time step showing the observed output interval
- **Intersection outlines**: Black lines where two scenarios' intervals overlap
- **Legend**: Scenario colours + intersection

### 2. Static Plot with Trajectory (`plot_refinement_3d_with_traj.py`)

- Same 4-panel layout as `plot_refinement_3d.py`
- **Trajectory overlay**: A single point-mass trajectory (Euler-integrated from `CarNomActSystem.f`) is drawn on all panels
- **CLI arguments**:
  - `--fault-mode {nominal, actuator, sensor}` — which fault mode the trajectory runs under
  - `--x0 px py phi` — initial state (default: `0.1 0.1 0.0`)
- **Trajectory simulation**: Uses the same `u_seq` and `dt` as the interval optimisation; for Actuator Fault, `alpha = 0.25` (midpoint of `[0.0, 0.5]`)

### 3. Animated Plot (`animate_refinement_3d.py`)

#### 3.1 Time Evolution

- **Forward reveal**: At frame `k`, only show interval boxes and trajectory up to time `t_k`. The plot grows forward as time advances.
- **Timing model**: `--sim-dt` (default `0.033` s) is the simulation timestep. It drives both the Euler integration and the interval propagation. Video FPS is derived as `round(1 / sim_dt)` — so the default gives ~30 fps. `--num-steps` controls the total number of frames.
- **Camera**: Fixed azimuth and elevation throughout (no sweep). Both configurable via `--azim` (default `45°`) and `--elev` (default `20°`).

#### 3.2 Zoom Strategy

One mechanism (azimuth sweep removed):

1. **Axis limits shrink** — At each frame, compute the bounding box of all *still-valid* interval boxes plus the trajectory points up to that frame. Apply a small padding (10% of range) and EMA-smooth the limits across frames (`α = 0.25`) to avoid jarring jumps.

#### 3.3 Validity Logic

A pair `(i, j)` is **still valid** at step `k` if the trajectory's observed output at step `k` falls inside **both** scenarios' refined output intervals at that step:

```
y_traj[k] = [px_traj[k] + δ_i, py_traj[k] + δ_i]   # scenario i's observed output
y_traj[k] ∈ obs_i[k] AND y_traj[k] ∈ obs_j[k]  →  pair is valid
```

Once a pair becomes invalid it stays invalid (no re-entry).

**Visual handling of invalid pairs**:
- When a pair becomes invalid, its interval boxes fade to transparent over `FADE_FRAMES = 3` frames
- After fading, the pair disappears entirely from the plot
- The title shows which pairs are still valid at each frame

#### 3.4 Output

- **Format**: `.mp4` via `matplotlib.animation.FuncAnimation` + `ffmpeg`
- **Resolution**: User-specified DPI (default: 150)
- **Filename**: `refinement_anim_{fault_mode}_simdt{sim_dt:.4f}.mp4` (or user-specified via `--output`)

---

## Usage

### Prerequisites

```bash
# ffmpeg must be installed
apt install ffmpeg        # Debian/Ubuntu
conda install ffmpeg      # conda
```

### Static Plot (no trajectory)

```bash
cd examples/faulty_car
python plot_refinement_3d.py
```

Output: `four_panel_refinement_dt0.5.pdf`

### Static Plot with Trajectory

```bash
python plot_refinement_3d_with_traj.py --fault-mode actuator --x0 0.1 0.1 0.0
```

**Arguments**:
- `--fault-mode {nominal, actuator, sensor}` — fault mode for trajectory (default: `nominal`)
- `--x0 px py phi` — initial state (default: `0.1 0.1 0.0`)

Output: `refinement_3d_traj_{fault_mode}_dt0.5.pdf`

### Animated Plot

```bash
python animate_refinement_3d.py --fault-mode actuator --x0 0.1 0.1 0.0
python animate_refinement_3d.py --fault-mode sensor --sim-dt 0.05 --dpi 200
python animate_refinement_3d.py --fault-mode nominal --azim 60 --elev 30
python animate_refinement_3d.py --fault-mode actuator --output my_video.mp4
```

**Arguments**:
- `--fault-mode {nominal, actuator, sensor}` — fault mode for trajectory (default: `nominal`)
- `--x0 px py phi` — initial state (default: `0.1 0.1 0.0`)
- `--sim-dt float` — simulation timestep in seconds (default: `0.033` → ~30 fps). Controls Euler integration, interval propagation, and video FPS (`round(1/sim_dt)`).
- `--num-steps int` — number of simulation steps (default: `300` → ~10 s at 0.033 s/step)
- `--azim float` — fixed camera azimuth in degrees (default: `45`)
- `--elev float` — fixed camera elevation in degrees (default: `20`)
- `--dpi int` — output video DPI (default: `150`)
- `--output path` — output .mp4 path (default: auto-named)

Output: `refinement_anim_{fault_mode}_simdt{sim_dt:.4f}.mp4`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Forward reveal** | Shows the evolution of the interval refinement process as the controller executes |
| **EMA-smoothed axis limits** | Prevents jarring jumps while still zooming in on relevant content |
| **Fixed camera angle** | Stable viewpoint; `--azim` and `--elev` let the user tune the angle without code changes |
| **Fade-out for invalid pairs** | Provides visual feedback that a pair was eliminated, not just abruptly removed |
| **sim-dt drives FPS** | Keeps simulation time and video time in sync; one parameter controls both |
| **Full redraw per frame** | Simpler than managing artists; correctness guaranteed |

---

## Example Output

### Static Plot (`plot_refinement_3d_with_traj.py`)

A 4-panel figure showing:
- Panel A: All three scenarios' unrefined output intervals
- Panels B–D: Refined intervals for each pair, with the trajectory overlaid

### Animated Plot (`animate_refinement_3d.py`)

A video showing:
- Interval boxes accumulating over time
- Trajectory line growing frame by frame
- Red dot marking the current measurement
- Axis limits shrinking to focus on relevant content
- Camera slowly rotating to follow the time front
- Title showing which pairs are still valid
- Invalid pairs fading out over 3 frames

---

## Implementation Notes

### Validity Computation

The validity matrix `valid[pair_idx, step_idx]` is pre-computed before animation:

```python
# For each pair (i, j) and step k:
y_traj[k] = traj[k, :2] + obs_offset_i
inside_i = (obs_i[k].lower[0] <= y_traj[k,0] <= obs_i[k].upper[0]) and ...
inside_j = (obs_j[k].lower[0] <= y_traj[k,0] <= obs_j[k].upper[0]) and ...
valid[pi, k] = inside_i and inside_j
```

Once `valid[pi, k]` becomes `False`, it stays `False` for all later steps.

### Bounding Box Computation

At each frame `k`, the bounding box includes:
- Trajectory points up to frame `k`
- Interval boxes for all valid pairs at steps `0..k-1`
- Interval boxes for pairs that are fading (within `FADE_FRAMES` of invalidation)

The bounding box is EMA-smoothed across frames to avoid jarring jumps.

### Camera Schedule

```python
azim[k] = AZIM_START + (AZIM_END - AZIM_START) * (k / num_frames)
```

This gives a smooth rotation from 45° to 20° as the animation progresses.
