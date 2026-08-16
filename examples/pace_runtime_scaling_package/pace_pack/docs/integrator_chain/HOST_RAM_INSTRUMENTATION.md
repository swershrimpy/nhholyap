# Coding instruction: add a per-order host RAM series to the integrator_chain sweep

Hand this file to a coding agent as its task specification. It is written to be
executed without further context-gathering: every file it names is in this
package, and every line anchor was checked against the current tree.

---

## Goal

The integrator_chain N=2..100 sweep currently records, per order N, a compile
time, a run time, and a peak **GPU** memory figure. Add a fourth per-order
series: **peak host (CPU) RAM**, so it can be plotted alongside the others.

## Why it does not already exist (read this before editing)

`_memory_snapshot()` in `code/integrator_chain/integrator_separating_input.py:214`
returns device stats **or** host RSS, never both:

```python
def _memory_snapshot() -> Dict[str, float]:
    """Best-effort memory snapshot: GPU device stats if available, else CPU RSS."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {k: float(v) for k, v in stats.items() if 'bytes' in k}   # <-- early return
    except Exception:
        pass
    return {'ru_maxrss_kb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
```

The sweep ran on a V100, so the early return fired every time and host RSS was
never sampled. **This cannot be recovered by re-parsing the existing log** --
the numbers were never printed. The only host figure job 11774929 produced is
the job-level Slurm epilog scalar `mem=24995472K` (~23.84 GiB peak, whole run),
in `logs/integrator_refinement_scaling_stable_11774929.out`. Collecting a
per-order series therefore requires a code change **and** a re-run on PACE
(~10h walltime).

## The measurement trap this task exists to avoid

Do **not** implement this with `resource.getrusage().ru_maxrss` or `/proc/self/status`
`VmHWM` alone. Both are *process-lifetime* high-water marks that can never go
down. Once order 90 has touched 20 GiB, every later order reports at least
20 GiB, so the series would be a running maximum, not a per-order measurement.

This is the same hazard the existing GPU series has -- `peak_bytes_in_use` is
also a process-lifetime running max (see `plot_refinement_runtime_scaling.py`'s
module docstring). That series is only usable because every one of the 99
orders happened to set a new record, which makes each value that order's own
peak. **Do not assume host RAM will be so lucky**: XLA compilation frees
buffers between orders, so host RSS can genuinely fall as N rises, and a
running-max series would silently hide that.

Sample a *resettable* peak instead: poll resident set size on an interval and
track the max since an explicit `reset()` at the top of each order.

---

## Edits

### 1. `code/integrator_chain/integrator_separating_input.py`

Insert above `_memory_snapshot()` (currently line 214, in the
`0. Timing / Memory Helper` section). `resource` is already imported; add
`os`, `threading`, `time` if absent.

```python
class _HostPeakRSS:
    """Per-order peak host RSS, sampled in a background thread.

    Reads field 2 (resident pages) of /proc/self/statm, which is cheaper than
    parsing /proc/self/status. Unlike ru_maxrss / VmHWM this peak is
    resettable, so it answers "how much host RAM did THIS order need" rather
    than "how much has the process ever touched" -- see this module's
    HOST_RAM_INSTRUMENTATION.md for why that distinction matters here.

    Linux-only, which is all PACE and the dev boxes are. On a platform without
    /proc, peak_bytes() returns 0.0 and the caller records an empty column
    rather than crashing.
    """

    def __init__(self, interval_s: float = 0.1):
        self._interval_s = interval_s
        self._peak = 0
        self._lock = threading.Lock()
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._started = False

    def _rss_bytes(self) -> int:
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * self._page
        except Exception:
            return 0

    def _poll_forever(self):
        while True:
            rss = self._rss_bytes()
            with self._lock:
                if rss > self._peak:
                    self._peak = rss
            time.sleep(self._interval_s)

    def start(self):
        """Idempotent -- safe to call more than once."""
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._poll_forever, daemon=True).start()

    def reset(self):
        """Drop the running peak to the CURRENT RSS. Call at the top of each order."""
        cur = self._rss_bytes()
        with self._lock:
            self._peak = cur

    def peak_bytes(self) -> float:
        with self._lock:
            return float(self._peak)


HOST_PEAK_RSS = _HostPeakRSS()
```

Then replace `_memory_snapshot()`'s body so it always returns **both** families
of keys:

```python
def _memory_snapshot() -> Dict[str, float]:
    """Best-effort memory snapshot: device stats AND host RSS -- both, always.

    Device keys keep their exact upstream names so existing log parsers and
    already-collected logs stay valid; host keys are added alongside.
    """
    snap: Dict[str, float] = {}
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            snap.update({k: float(v) for k, v in stats.items() if 'bytes' in k})
    except Exception:
        pass
    snap['host_peak_rss_bytes'] = HOST_PEAK_RSS.peak_bytes()
    # ru_maxrss is KiB on Linux; kept as the monotone process-lifetime
    # reference, used only to cross-check the final value against Slurm's
    # job-level `mem=` figure. It is NOT the per-order series.
    snap['host_ru_maxrss_bytes'] = float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0
    return snap
```

**Constraints:**
- Do not rename or drop any existing `*bytes*` device key -- `peak_bytes_in_use`
  in particular is what the committed parser matches on.
- The old `'ru_maxrss_kb'` key disappears. `grep -rn "ru_maxrss_kb"` across the
  repo first; at time of writing nothing consumes it, but each example module
  has its own copy of `time_jit`/`_memory_snapshot`, so confirm rather than assume.
- Do not change `time_jit`'s signature (`integrator_separating_input.py:225`,
  returns `(jitted_fn, compile_time_s, run_time_s, memory_info)`); it already
  calls `_memory_snapshot()` at line 244 and needs no edit.

### 2. `code/integrator_chain/refinement_demo_stable.py`

- Add `HOST_PEAK_RSS` to the `from integrator_separating_input import (...)`
  block (lines 80-93).
- Call `HOST_PEAK_RSS.start()` once in `__main__`, before the
  `for N in range(2, 101):` loop at line 262.
- Call `HOST_PEAK_RSS.reset()` as the **first statement** of `run_for_order`
  (line 219).

Placing the reset at the top of `run_for_order` -- not inside `time_jit` --
makes the recorded peak cover that order's whole body, including the unrefined
optimizer that runs before the timed call. That matches the scope of the
existing GPU number and is the more useful figure, since host RAM here is
dominated by XLA compilation.

Leave the `print(f"\nMemory snapshot: {mem}")` at line 257 exactly as is. The
new keys ride inside the same dict on the same one-line format, so the existing
`Memory snapshot: {...}` regex contract is preserved.

### 3. `code/integrator_chain/plot_refinement_runtime_scaling.py`

- Add next to `_PEAK_MEM_RE` (line 76):
  ```python
  _HOST_RSS_RE = re.compile(r"'host_peak_rss_bytes': ([\d.eE+-]+)")
  ```
- Widen `parse_log` (line 79) rows from 4-tuples to 5-tuples, appending
  `peak_host_rss_bytes`. Follow the existing `peak_gpu_mem_bytes` handling
  exactly: `None` when the key is absent, so the **already-committed log
  still parses** and simply yields an empty column.
- Add `"peak_host_rss_bytes"` to the header and rows in `write_csv` (line 115),
  reusing the same `"" if m is None else f"{m:.0f}"` formatting.
- In `__main__` (from line 162), print the host min/max the same way the GPU
  min/max is printed. **Do not** copy the running-max/censoring warning onto the
  host series -- it is reset per order, so a flat or falling stretch is a real
  measurement, not a censored one. Copying that warning would be actively wrong.

### 4. `code/integrator_chain/plot_post_compile_runtime_filtered.py`

Host RAM (~23.8 GiB) and GPU memory (~148 MB) are ~160x apart. **Do not add it
as a third curve on the existing twin axis** -- it would flatten the GPU series
into the baseline, and the file's own docstring already records that a twin
axis is the one form the project's dataviz guidance rules out by default.

Instead:
- Extend `load_rows` (line 120) to return the host column too, keeping the
  existing "column absent -> `None`, handled by the caller" contract at line 129.
- Give host RAM its **own panel**: extend the `twin_axis=False` branch of
  `make_plot` (line 195) from two panels to three (run time | GPU memory |
  host RAM), and leave the default twin-axis figure as the two-series plot it
  is today.
- Use categorical slot 3 (aqua `#1baf7a`) for the host series, keeping slot 1
  blue = run time and slot 2 orange = GPU memory. Validate before shipping:
  `node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light`
  from the `dataviz` skill directory; all six checks must PASS.
- Run the existing local-median filter over the host series in its own units
  (GiB), as is already done for the other two, and report exclusions in
  `__main__`. Plot every surviving point connected by lines -- no `markevery`
  subsampling; a sparse-marker memory curve reads as fitted rather than measured.

### 5. Re-run on PACE

`code/integrator_chain/run_refinement_scaling_stable.sbatch` needs no resource
change: it already requests `--cpus-per-task=8 --mem=80G --gres=gpu:v100:1`,
and the measured host peak was 23.84 GiB, so 80G has ample headroom.

Before submitting, check `scontrol show reservation` for a maintenance window.
That sbatch's header documents the trap: requesting a walltime whose worst case
does not fit before a reservation makes Slurm silently defer the job until
after the maintenance ends rather than refuse it. The previous run took 9h49m
against a `--time=1-02:00:00` request.

---

## Acceptance criteria

1. **Old log still parses.** `python plot_refinement_runtime_scaling.py ../../logs/integrator_refinement_scaling_stable_11774929.out`
   produces 99 rows, all 4 existing columns unchanged and byte-identical to the
   committed CSV, with `peak_host_rss_bytes` empty. No crash, no exception.
2. **Local smoke test.** Temporarily narrow the loop to `range(2, 6)` and run
   with `JAX_PLATFORMS=cpu`. Every `Memory snapshot:` line must contain
   `host_peak_rss_bytes` and a non-zero value. On CPU there are no device keys
   at all -- confirm the parser handles that (GPU column empty, host column
   populated), which is the mirror image of case 1.
3. **The reset actually works.** In a GPU run, `host_peak_rss_bytes` must not be
   a monotone non-decreasing sequence identical in shape to
   `host_ru_maxrss_bytes`. If the two series match exactly, `reset()` is not
   being called per order and the number is a process-lifetime max -- the exact
   bug this task exists to prevent. Fail the task in that case.
4. **External cross-check.** `max(host_ru_maxrss_bytes)` over all orders must
   land within ~10% of the Slurm epilog's job-level `mem=` figure for the same
   job. Two independent measurements of the same quantity; if they disagree by
   much more, the instrumentation is wrong.
5. **Sampler overhead is negligible.** Post-compilation run times must stay in
   the same band as the committed CSV (~3 ms at low N to ~24 ms at N=100). A
   0.1s poll of a small `/proc` read against 60-190s compiles should be
   unmeasurable; if run times move materially, raise the interval.
6. **Palette validated** for the 3-slot categorical set, per edit 4.

## Out of scope

- Changing `time_jit`'s signature or its two-call compile-vs-run structure.
- Touching the other examples' copies of `_memory_snapshot`/`time_jit`
  (`code/admire/`, `code/nonlinear_chain/`). They are independent copies; leave
  them alone unless the same series is wanted there, which is a separate task.
- Re-deriving the GPU series. It is correct as committed.
