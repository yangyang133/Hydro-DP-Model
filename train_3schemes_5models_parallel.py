"""Parallel batch training runner (parallel over case+model tasks).

Task granularity:
- each task is one (scheme, case, model)
- tasks run in parallel as independent subprocesses

This wrapper reuses train_3schemes_5models.py for the actual task execution.
It forwards the refactored param-mode options to each subprocess.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import train_3schemes_5models as serial_runner


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel train: 3 schemes x 5 models")
    parser.add_argument("--models", nargs="+", default=None, help="Subset models, e.g. XAJ GR4J SAC-SMA Tank HBV")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=None,
        help="Subset schemes, e.g. scheme1_separate scheme2_cross scheme3_joint",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Subset case names, e.g. js_only train_js_valid_hhy",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument(
        "--summary-file",
        type=str,
        default="results/three_schemes/train_summary_parallel.csv",
        help="Single merged summary CSV path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/three_schemes",
        help="Root directory for per-run outputs",
    )
    parser.add_argument("--max-workers", type=int, default=6, help="Max parallel workers")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--keep-temp", action="store_true", help="Keep per-task temporary summaries")
    parser.add_argument(
        "--param-category",
        type=str,
        default="dl",
        choices=["dl", "traditional"],
        help="Top-level parameter mode category",
    )
    parser.add_argument(
        "--param-strategy",
        type=str,
        default=None,
        help="Mode strategy. dl: scratch/continue. traditional: manual/differential_evolution",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Root directory for dl/continue checkpoints",
    )
    parser.add_argument("--de-maxiter", type=int, default=None, help="Override DE max iterations")
    parser.add_argument("--de-popsize", type=int, default=None, help="Override DE population size")
    parser.add_argument("--de-mutation", type=float, default=None, help="Override DE mutation factor")
    parser.add_argument("--de-recombination", type=float, default=None, help="Override DE crossover rate")
    parser.add_argument("--de-tol", type=float, default=None, help="Override DE early-stop tolerance")
    parser.add_argument(
        "--write-to-data-params",
        action="store_true",
        help="For traditional differential evolution, also copy best params to data/<basin>/params",
    )
    return parser.parse_args()


def _make_task_list(selected_models, selected_schemes, mode_category, mode_strategy, output_root):
    tasks = []
    for scheme_name, cases in selected_schemes.items():
        for case in cases:
            for model_info in selected_models:
                tasks.append(
                    {
                        "scheme": scheme_name,
                        "case": case["name"],
                        "case_desc": case.get("description", ""),
                        "model": model_info["name"],
                        "param_category": mode_category,
                        "param_strategy": mode_strategy,
                        "output_root": output_root,
                    }
                )
    return tasks


def _build_command(args, task, temp_summary):
    cmd = [
        args.python,
        "train_3schemes_5models.py",
        "--schemes",
        task["scheme"],
        "--cases",
        task["case"],
        "--models",
        task["model"],
        "--summary-file",
        str(temp_summary),
        "--param-category",
        args.param_category,
    ]
    if args.param_strategy is not None:
        cmd.extend(["--param-strategy", args.param_strategy])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(int(args.epochs))])
    if args.output_root:
        cmd.extend(["--output-root", args.output_root])
    if args.checkpoint_root:
        cmd.extend(["--checkpoint-root", args.checkpoint_root])
    if args.de_maxiter is not None:
        cmd.extend(["--de-maxiter", str(int(args.de_maxiter))])
    if args.de_popsize is not None:
        cmd.extend(["--de-popsize", str(int(args.de_popsize))])
    if args.de_mutation is not None:
        cmd.extend(["--de-mutation", str(float(args.de_mutation))])
    if args.de_recombination is not None:
        cmd.extend(["--de-recombination", str(float(args.de_recombination))])
    if args.de_tol is not None:
        cmd.extend(["--de-tol", str(float(args.de_tol))])
    if args.write_to_data_params:
        cmd.append("--write-to-data-params")
    return cmd


def _run_one_task(task, args, temp_dir, log_dir):
    scheme = task["scheme"]
    case = task["case"]
    model = str(task["model"]).lower()

    key = "{}__{}__{}".format(scheme, case, model)
    temp_summary = temp_dir / "{}.csv".format(key)
    log_file = log_dir / "{}.log".format(key)

    cmd = _build_command(args, task, temp_summary)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n{}\n\n".format(" ".join(cmd)))
        f.write("RETURN_CODE: {}\n\n".format(proc.returncode))
        f.write("STDOUT:\n")
        f.write(proc.stdout or "")
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr or "")

    return {
        "task": task,
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "temp_summary": str(temp_summary),
        "log_file": str(log_file),
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def _failure_row(task, error_msg):
    model_lower = str(task["model"]).lower()
    base = Path(task.get("output_root", "results/three_schemes")) / task["scheme"] / task["case"] / model_lower / "train"
    return {
        "scheme": task["scheme"],
        "case": task["case"],
        "case_desc": task.get("case_desc", ""),
        "model": task["model"],
        "param_category": task.get("param_category", ""),
        "param_strategy": task.get("param_strategy", ""),
        "checkpoint_dir": "",
        "status": "failed",
        "best_valid_kge": float("nan"),
        "final_train_kge": float("nan"),
        "total_epochs": 0,
        "time_seconds": float("nan"),
        "output_dir": str(base).replace("\\", "/"),
        "generated_config": str(base / "generated_config.yaml").replace("\\", "/"),
        "error": error_msg,
    }


def main():
    args = parse_args()
    selected_models = serial_runner.select_model_configs(args.models)
    selected_schemes = serial_runner.select_schemes(args.schemes, args.cases)
    mode_category, mode_strategy = serial_runner.normalize_mode_strategy(args.param_category, args.param_strategy)
    tasks = _make_task_list(selected_models, selected_schemes, mode_category, mode_strategy, args.output_root)

    if not tasks:
        raise ValueError("No task selected")

    max_workers = args.max_workers
    if max_workers is None or max_workers <= 0:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    else:
        max_workers = min(max_workers, len(tasks))

    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = summary_path.parent / "_parallel_tmp"
    log_dir = summary_path.parent / "_parallel_logs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Parallel training started")
    print("Task count: {}".format(len(tasks)))
    print("Workers: {}".format(max_workers))
    print("Models: {}".format(", ".join(m["name"] for m in selected_models)))
    print("Schemes: {}".format(", ".join(selected_schemes.keys())))
    print("Param mode: {}/{}".format(mode_category, mode_strategy))
    print("=" * 100)

    t_all = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_task, task, args, temp_dir, log_dir): task for task in tasks}
        for fut in as_completed(futures):
            out = fut.result()
            task = out["task"]
            label = "{} / {} / {} / {}/{}".format(
                task["scheme"],
                task["case"],
                task["model"],
                task["param_category"],
                task["param_strategy"],
            )
            if out["returncode"] == 0:
                print("[Done] {} | {:.1f} min".format(label, out["elapsed"] / 60.0))
            else:
                print("[Fail] {} | rc={} | see {}".format(label, out["returncode"], out["log_file"]))
            results.append(out)

    merged_frames = []
    failed_rows = []
    for out in results:
        task = out["task"]
        temp_summary = Path(out["temp_summary"])
        if out["returncode"] != 0:
            failed_rows.append(_failure_row(task, "subprocess rc={}".format(out["returncode"])))
            continue

        if not temp_summary.exists():
            failed_rows.append(_failure_row(task, "missing temp summary"))
            continue

        try:
            df = pd.read_csv(temp_summary)
        except Exception as exc:
            failed_rows.append(_failure_row(task, "read temp summary failed: {}".format(exc)))
            continue

        if "model" in df.columns:
            df = df[df["model"].astype(str).str.lower() == str(task["model"]).lower()]
        merged_frames.append(df)

    if merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
    else:
        merged_df = pd.DataFrame()

    if failed_rows:
        merged_df = pd.concat([merged_df, pd.DataFrame(failed_rows)], ignore_index=True)

    if not merged_df.empty:
        sort_cols = [c for c in ["scheme", "case", "model"] if c in merged_df.columns]
        if sort_cols:
            merged_df = merged_df.sort_values(sort_cols, ascending=[True] * len(sort_cols))

    merged_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not args.keep_temp:
        for p in temp_dir.glob("*.csv"):
            p.unlink(missing_ok=True)

    elapsed_all = time.time() - t_all
    print("=" * 100)
    print("Parallel training finished")
    print("Total time: {:.1f} min ({:.2f} h)".format(elapsed_all / 60.0, elapsed_all / 3600.0))
    print("Summary CSV: {}".format(summary_path))
    print("Task logs: {}".format(log_dir))
    print("=" * 100)


if __name__ == "__main__":
    main()
