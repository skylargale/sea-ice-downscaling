"""
rename_results.py

One-off script: renames Version4/results/<run>/ folders to be self-
describing (mode, train years, test years, job id), using the same
sources label_results.py reads from (run_config.json when present,
otherwise eval_data/meta.json, sit_timeseries.csv, and the old folder
name). Also patches the "output_dir" field inside any run_config.json
it moves, and the one hardcoded path in
engressnet_evaluation_plots.ipynb (EVAL_DIR -> FOSI_5185183.casper-pbs).

New name format: "<mode>_train<TRAIN>_test<TEST>_<jobid>", e.g.
    nopatches_train1958-2000_test2001-2022_5206874

Safe to run only when no PBS job is currently writing into results/
(check `qstat -u $USER` first) -- confirmed by hand before running this
once; not re-run automatically.
"""

import json
import os
import re

import label_results as lr

RESULTS_DIR = lr.DEFAULT_RESULTS_DIR
NOTEBOOK_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "engressnet_evaluation_plots.ipynb")


def condense_years(years):
    """[1958, 1959, ..., 2000] -> '1958-2000'; non-contiguous -> '1958-1960,1965'."""
    if not years:
        return None
    ys = sorted(years)
    ranges = []
    start = prev = ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
            continue
        ranges.append((start, prev))
        start = prev = y
    ranges.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def extract_jobid(name):
    m = re.search(r"(\d+)\.casper", name)
    return m.group(1) if m else name


def plan_new_name(name, run_dir):
    jobid = extract_jobid(name)
    config_path = os.path.join(run_dir, "run_config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        mode = "patches" if cfg.get("use_patches") else "nopatches"
        train = condense_years(cfg.get("train_years")) or "randomsplit"
        test = condense_years(cfg.get("test_years")) or "randomsplit"
        return f"{mode}_train{train}_test{test}_{jobid}"

    meta = lr.read_eval_meta(run_dir)
    ts_range = lr.read_timeseries_year_range(run_dir)
    dirname_train, dirname_test = lr.parse_years_from_dirname(name)
    has_timeseries = os.path.exists(os.path.join(run_dir, "sit_timeseries.csv"))

    if meta is not None:
        mode = "nopatches" if not meta.get("use_patches") else "patches"
    elif has_timeseries:
        mode = "nopatches"
    else:
        mode = "patches-unconfirmed"

    train = dirname_train or "unknown"
    test = dirname_test or ts_range or "unknown"
    return f"{mode}_train{train}_test{test}_{jobid}"


def main():
    renames = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(run_dir) or name.startswith(".") or name == "results":
            continue
        new_name = plan_new_name(name, run_dir)
        if new_name != name:
            renames[name] = new_name

    print("Planned renames:")
    for old, new in renames.items():
        print(f"  {old}  ->  {new}")

    for old, new in renames.items():
        old_path = os.path.join(RESULTS_DIR, old)
        new_path = os.path.join(RESULTS_DIR, new)
        os.rename(old_path, new_path)

        config_path = os.path.join(new_path, "run_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            cfg["output_dir"] = new_path
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2, default=str)

    if os.path.exists(NOTEBOOK_PATH) and "FOSI_5185183.casper-pbs" in renames:
        new_eval_ref = renames["FOSI_5185183.casper-pbs"]
        with open(NOTEBOOK_PATH) as f:
            text = f.read()
        updated = text.replace("FOSI_5185183.casper-pbs", new_eval_ref)
        if updated != text:
            with open(NOTEBOOK_PATH, "w") as f:
                f.write(updated)
            print(f"\nUpdated engressnet_evaluation_plots.ipynb EVAL_DIR reference -> {new_eval_ref}")

    print(f"\nDone. Renamed {len(renames)} folders.")


if __name__ == "__main__":
    main()
