"""
Recovery tool: rebuild a method's summary CSV from whatever per-horizon
.npz files it already managed to write before dying.

Why this exists: admire_success_rate_analysis.py's main() writes its
summary CSV once, at the very end, only after every horizon for that
invocation has finished (see main()'s comment for the incremental-write
fix this predates for *future* runs). A run that gets killed mid-sweep
(OOM, walltime, a cluster-maintenance reclaim) before that final write
still has each already-completed horizon's raw data safely on disk --
run_method_for_horizon saves admire_success_rate_data_<method>_<horizon>s.npz
right after that horizon finishes, independent of the final CSV write.
This script reconstructs exactly the summary rows main() would have
written, from those .npz files alone, so a killed run isn't a total loss.

Usage
-----
    python rebuild_summary_csv_from_npz.py --method single_step
    python rebuild_summary_csv_from_npz.py --method single_step --out /tmp/recovered.csv

Only reconstructs whichever (method, horizon) .npz files are present --
missing horizons are simply absent from the output, with a note printed
to stderr about what was and wasn't found.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from admire_success_rate_analysis import (
    HORIZONS_S, _METHOD_LOSS_BUILDERS, _METHOD_DISPLAY, _out_csv_for_method,
)


def rebuild(method: str, out_csv: Path):
    rows = []
    found, missing = [], []

    for horizon_s in HORIZONS_S:
        npz_path = _HERE / f"admire_success_rate_data_{method}_{horizon_s}s.npz"
        if not npz_path.exists():
            missing.append(horizon_s)
            continue
        found.append(horizon_s)

        data = np.load(npz_path)
        losses_all = data["losses_final_all_restarts"]      # (num_configs, num_restarts)
        config_success = data["config_success"]              # (num_configs,)
        restart_success_rate = data["restart_success_rate"]  # (num_configs,)
        input_limit = data["input_limit"]                    # (num_configs,)
        x0_width = data["x0_width"]                          # (num_configs,)

        best_loss = np.min(losses_all, axis=1)   # same value argmin would have selected
        name = _METHOD_DISPLAY[method]

        for c in range(losses_all.shape[0]):
            rows.append({
                'method': name, 'horizon_s': horizon_s, 'config_idx': c,
                'input_limit': float(input_limit[c]), 'x0_width': float(x0_width[c]),
                'success': bool(config_success[c]), 'final_loss': float(best_loss[c]),
                'restart_success_rate': float(restart_success_rate[c]),
            })

    if not rows:
        print(f"No .npz files found for method={method!r} in {_HERE} -- nothing to rebuild.",
              file=sys.stderr)
        sys.exit(1)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rebuilt {len(rows)} rows ({len(found)}/{len(HORIZONS_S)} horizons: {found}) -> {out_csv}")
    if missing:
        print(f"Missing horizons (no .npz found, likely not reached before the run ended): {missing}",
              file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(_METHOD_LOSS_BUILDERS))
    parser.add_argument("--out", type=Path, default=None,
                        help="Defaults to the same path main() would have used "
                             "(admire_success_rate_summary_<method>.csv in this directory).")
    args = parser.parse_args()
    rebuild(args.method, args.out or _out_csv_for_method(args.method))
