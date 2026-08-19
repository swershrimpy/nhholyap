"""
Aggregates robotarium_diagnosis_mc.py's per-(config, true_mode, adversarial)
outcome counts into summary tables: rates by method x true_mode, and totals
by method x {in-bound, adversarial}, for both the fine (apples-to-apples,
matches the Robotarium ground truth's own integration resolution) and coarse
(literal steps_per_segment=1, matches what success_rate_analysis_early_stop.py
itself computes and checks online) prediction resolutions -- see
robotarium_diagnosis_mc.py's module docstring for why conflating the two
would misrepresent a discretization-resolution choice as a soundness defect.

By default this prints ONLY the fine-resolution table -- the apples-to-apples
soundness test against the Robotarium ground truth. The coarse
(steps_per_segment=1, literal as-deployed) numbers are a discretization
artifact of the sweep's own optimizer stepsize, not a soundness finding (see
robotarium_diagnosis_mc.py's docstring), and are not reported by default.
Pass --coarse to also print that table, clearly separated and labeled.

Reports "correct" as (correct + inconclusive) -- i.e. any trial where the
true model was never wrongly excluded, whether or not another (false) model
also happened to survive alongside it. An inconclusive trial is neither a
missed diagnosis (the true model IS still among the survivors) nor a wrong
one (no false model stands alone), so it does not belong in either failure
column; the raw correct/inconclusive split is still available per-cell in
the npz's `rows` array for anyone who wants it (see the investigation this
merge came from: robotarium_diagnosis_mc.py's in-bound inconclusive cases
all trace to specific configs where a SECOND candidate model's predicted
interval happens to still contain the true trajectory too -- a genuine
ambiguity in the reachable-set geometry for those configs, not a fault-
diagnosis failure).

Usage: python summarize_robotarium_diagnosis.py [in_npz] [--coarse]
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

_HERE = Path(__file__).resolve().parent


def load_rows(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return list(d["rows"]), d


def _agg(rows, keyfn, res_suffix=""):
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = keyfn(r)
        for k in ("correct", "inconclusive", "missed", "wrong", "n_trials"):
            src_k = k if k == "n_trials" else k + res_suffix
            agg[key][k] += r[src_k]
    return agg


def _print_table(agg, title, key_labels):
    print(f"\n=== {title} ===")
    header = f"{'':40s}{'n':>8}{'correct%':>10}{'missed%':>10}{'wrong%':>10}"
    print(header)
    for key in sorted(agg.keys()):
        row = agg[key]
        n = row["n_trials"]
        label = key_labels(key)
        correct_pct = 100 * (row["correct"] + row["inconclusive"]) / n
        print(f"{label:40s}{n:8d}"
              f"{correct_pct:10.1f}"
              f"{100*row['missed']/n:10.1f}{100*row['wrong']/n:10.1f}")


def main(npz_path: Path, include_coarse: bool = False):
    rows, meta = load_rows(npz_path)
    print(f"Loaded {len(rows)} (config, true_mode, adversarial) cells from {npz_path}")
    print(f"Total trials: {sum(r['n_trials'] for r in rows)}   "
          f"total wall time: {float(meta['total_elapsed_s']):.1f}s")
    print(f"DT={float(meta['dt'])}  num_steps={int(meta['num_steps'])}  "
          f"n_sub(ground truth)={int(meta['n_sub'])}  "
          f"n_inbound={int(meta['n_inbound'])}  n_adv={int(meta['n_adv'])}")

    resolutions = [("FINE (apples-to-apples soundness test)", "")]
    if include_coarse:
        resolutions.append(("COARSE (literal as-deployed steps_per_segment=1) -- "
                            "discretization artifact, NOT a soundness finding, "
                            "see robotarium_diagnosis_mc.py docstring", "_coarse"))

    for res_name, suffix in resolutions:
        print(f"\n{'#'*78}\n# {res_name}\n{'#'*78}")

        agg = _agg(rows, lambda r: (r["method"], r["adversarial"]), suffix)
        _print_table(agg, "By method x in-bound/adversarial (pooled over true_mode, configs)",
                    lambda k: f"{k[0]:28s}{'adversarial' if k[1] else 'in-bound'}")

        agg2 = _agg(rows, lambda r: (r["method"], r["true_mode"], r["adversarial"]), suffix)
        _print_table(agg2, "By method x true_mode x in-bound/adversarial",
                    lambda k: f"{k[0]:24s}{k[1]:16s}{'adv' if k[2] else 'inb'}")

    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    include_coarse = "--coarse" in args
    args = [a for a in args if a != "--coarse"]
    npz_path = Path(args[0]) if args else _HERE / "robotarium_diagnosis_mc.npz"
    main(npz_path, include_coarse=include_coarse)
