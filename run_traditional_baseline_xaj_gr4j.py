from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import train as train_module
import train_3schemes_5models as batch_runner


DEFAULT_MODEL_CONFIGS = [
    {"name": "XAJ", "config_file": "configs/config_xaj_cold.yaml"},
    {"name": "GR4J", "config_file": "configs/config_gr4j_cold.yaml"},
]

DEFAULT_BASINS = ["js", "hhy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traditional baseline runner for XAJ/GR4J over Yangtze and Yellow headwaters"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["XAJ", "GR4J"],
        help="Subset models. Defaults to: XAJ GR4J",
    )
    parser.add_argument(
        "--basins",
        nargs="+",
        default=["js", "hhy"],
        help="Subset basins. Defaults to: js hhy",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/traditional_baseline_xaj_gr4j",
        help="Root directory for all baseline outputs",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Optional summary CSV path. Defaults to <output-root>/baseline_summary.csv",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(number of runs, CPU count)",
    )
    parser.add_argument("--de-maxiter", type=int, default=200, help="DE max iterations")
    parser.add_argument("--de-popsize", type=int, default=15, help="DE population size")
    parser.add_argument("--de-mutation", type=float, default=0.8, help="DE mutation factor")
    parser.add_argument("--de-recombination", type=float, default=0.7, help="DE crossover rate")
    parser.add_argument("--de-tol", type=float, default=0.01, help="DE early-stop tolerance")
    parser.add_argument("--de-seed", type=int, default=42, help="DE random seed")
    parser.add_argument(
        "--write-to-data-params",
        action="store_true",
        help="Also copy calibrated params to data/<basin>/params",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def normalize_basin_names(requested_basins):
    basin_map = {
        "js": "js",
        "jinshang": "js",
        "yangtze": "js",
        "changjiang": "js",
        "hhy": "hhy",
        "huangheyuan": "hhy",
        "yellow": "hhy",
        "huanghe": "hhy",
    }
    normalized = []
    for item in requested_basins or []:
        key = str(item).strip().lower()
        if key not in basin_map:
            raise ValueError(f"Unknown basin: {item}")
        basin_name = basin_map[key]
        if basin_name not in normalized:
            normalized.append(basin_name)
    if not normalized:
        raise ValueError("No basin selected.")
    return normalized


def select_model_configs(requested_models):
    if not requested_models:
        return copy.deepcopy(DEFAULT_MODEL_CONFIGS)

    normalized_to_name = {
        "xaj": "XAJ",
        "gr4j": "GR4J",
    }
    requested = []
    for item in requested_models:
        key = str(item).strip().lower().replace("-", "").replace("_", "")
        if key not in normalized_to_name:
            raise ValueError(f"Unknown model: {item}")
        canonical = normalized_to_name[key]
        if canonical not in requested:
            requested.append(canonical)
    return [m for m in DEFAULT_MODEL_CONFIGS if m["name"] in requested]


def make_baseline_case(basin_name: str) -> dict:
    basin_cfg = batch_runner.make_basin(basin_name, role="both")
    description_map = {
        "js": "Traditional DE baseline in Yangtze headwaters",
        "hhy": "Traditional DE baseline in Yellow headwaters",
    }
    return {
        "name": f"{basin_name}_baseline",
        "description": description_map.get(basin_name, basin_name),
        "basins": [basin_cfg],
    }


def enforce_baseline_settings(config: dict, basin_name: str, output_dir: Path, args) -> dict:
    config = copy.deepcopy(config)
    config["basins"] = [batch_runner.make_basin(basin_name, role="both")]

    model_cfg = config.setdefault("model", {})
    cold_cfg = model_cfg.setdefault("cold_region", {})
    cold_cfg["enabled"] = True

    precip_cfg = model_cfg.setdefault("precip_correct", {})
    precip_cfg["enabled"] = True
    precip_cfg.setdefault("kp", 1.0)

    components = model_cfg.setdefault("components", {})
    active_components = train_module.get_active_component_names(config)
    for component_name in active_components:
        component_cfg = components.setdefault(component_name, {})
        component_cfg["param_mode"] = {
            "category": "traditional",
            "strategy": "differential_evolution",
        }

    training_cfg = config.setdefault("training", {})
    training_cfg["output_dir"] = str(output_dir).replace("\\", "/")
    training_cfg.pop("resume", None)
    traditional_cfg = copy.deepcopy(training_cfg.get("traditional", {}))
    traditional_cfg.update(
        {
            "maxiter": int(args.de_maxiter),
            "popsize": int(args.de_popsize),
            "mutation": float(args.de_mutation),
            "recombination": float(args.de_recombination),
            "tol": float(args.de_tol),
            "seed": int(args.de_seed),
            "write_to_data_params": bool(args.write_to_data_params),
        }
    )
    training_cfg["traditional"] = traditional_cfg
    return config


def read_calibration_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "calibration_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_max_workers(requested: int | None, total_runs: int) -> int:
    if total_runs <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(requested), total_runs))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, total_runs))


def write_summary_csv(results: list[dict], summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["model", "basin"], ascending=[True, True])
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")


def run_one_baseline(model_info: dict, basin_name: str, args) -> dict:
    # Worker processes need to re-apply the gauge selection patch after spawn.
    batch_runner.patch_observation_loader_for_selected_gauges()

    base_config = load_yaml(model_info["config_file"])
    model_tag = batch_runner.model_output_tag(model_info["name"])
    output_dir = Path(args.output_root) / model_tag / basin_name / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = enforce_baseline_settings(
        config=base_config,
        basin_name=basin_name,
        output_dir=output_dir,
        args=args,
    )

    generated_config_path = output_dir / "generated_config.yaml"
    save_yaml(generated_config_path, config)

    started = time.time()
    error_msg = ""
    status = "success"

    try:
        train_module.train(config, config_path=str(generated_config_path))
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        traceback.print_exc()

    elapsed = time.time() - started
    calibration_summary = read_calibration_summary(output_dir) if status == "success" else {}

    optimization = calibration_summary.get("optimization_result", {})
    train_summary = calibration_summary.get("train", {})
    valid_summary = calibration_summary.get("valid", {})

    return {
        "model": model_info["name"],
        "basin": basin_name,
        "status": status,
        "cold_region_enabled": True,
        "precip_correct_enabled": True,
        "de_maxiter": int(args.de_maxiter),
        "de_popsize": int(args.de_popsize),
        "de_mutation": float(args.de_mutation),
        "de_recombination": float(args.de_recombination),
        "de_tol": float(args.de_tol),
        "de_seed": int(args.de_seed),
        "best_objective": float(optimization.get("best_objective", np.nan)),
        "iterations_completed": int(optimization.get("iterations_completed", 0)) if optimization else 0,
        "function_evaluations": int(optimization.get("function_evaluations", 0)) if optimization else 0,
        "objective_train": float(train_summary.get("objective", np.nan)),
        "objective_valid": float(valid_summary.get("objective", np.nan)),
        "avg_kge_train": float(train_summary.get("avg_kge", np.nan)),
        "avg_kge_valid": float(valid_summary.get("avg_kge", np.nan)),
        "time_seconds": float(elapsed),
        "output_dir": str(output_dir).replace("\\", "/"),
        "generated_config": str(generated_config_path).replace("\\", "/"),
        "error": error_msg,
    }


def format_result_message(result: dict) -> str:
    if result["status"] == "success":
        return (
            "  Success | train_kge={:.4f} | valid_kge={:.4f} | best_objective={:.4f} "
            "| iters={} | time={:.1f} min"
        ).format(
            result["avg_kge_train"],
            result["avg_kge_valid"],
            result["best_objective"],
            result["iterations_completed"],
            result["time_seconds"] / 60.0,
        )
    return "  Failed | error={}".format(result["error"])


def main():
    args = parse_args()
    batch_runner.patch_observation_loader_for_selected_gauges()

    selected_models = select_model_configs(args.models)
    selected_basins = normalize_basin_names(args.basins)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file) if args.summary_file else output_root / "baseline_summary.csv"

    run_cases = [
        {
            "index": index,
            "model_info": model_info,
            "basin_name": basin_name,
        }
        for index, (model_info, basin_name) in enumerate(
            [(m, b) for m in selected_models for b in selected_basins],
            start=1,
        )
    ]
    total_runs = len(run_cases)
    max_workers = resolve_max_workers(args.max_workers, total_runs)

    print("=" * 100, flush=True)
    print("Traditional baseline started", flush=True)
    print("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    print("Models: {}".format(", ".join(m["name"] for m in selected_models)), flush=True)
    print("Basins: {}".format(", ".join(selected_basins)), flush=True)
    print("Mode: traditional/differential_evolution", flush=True)
    print("Cold region: enabled", flush=True)
    print("Precipitation correction: enabled", flush=True)
    print(
        "DE settings: maxiter={} | popsize={} | mutation={} | recombination={} | tol={} | seed={}".format(
            args.de_maxiter,
            args.de_popsize,
            args.de_mutation,
            args.de_recombination,
            args.de_tol,
            args.de_seed,
        ),
        flush=True,
    )
    print("Parallel workers: {}".format(max_workers), flush=True)
    print("=" * 100, flush=True)

    all_results = []
    total_started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one_baseline, case["model_info"], case["basin_name"], args): case
            for case in run_cases
        }
        for future in concurrent.futures.as_completed(future_map):
            case = future_map[future]
            model_name = case["model_info"]["name"]
            basin_name = case["basin_name"]
            print(
                "\n[{} / {}] {} | {} | traditional/differential_evolution".format(
                    case["index"],
                    total_runs,
                    model_name,
                    basin_name,
                ),
                flush=True,
            )
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "model": model_name,
                    "basin": basin_name,
                    "status": "failed",
                    "cold_region_enabled": True,
                    "precip_correct_enabled": True,
                    "de_maxiter": int(args.de_maxiter),
                    "de_popsize": int(args.de_popsize),
                    "de_mutation": float(args.de_mutation),
                    "de_recombination": float(args.de_recombination),
                    "de_tol": float(args.de_tol),
                    "de_seed": int(args.de_seed),
                    "best_objective": float(np.nan),
                    "iterations_completed": 0,
                    "function_evaluations": 0,
                    "objective_train": float(np.nan),
                    "objective_valid": float(np.nan),
                    "avg_kge_train": float(np.nan),
                    "avg_kge_valid": float(np.nan),
                    "time_seconds": float(np.nan),
                    "output_dir": str(
                        (output_root / batch_runner.model_output_tag(model_name) / basin_name / "train")
                    ).replace("\\", "/"),
                    "generated_config": str(
                        (
                            output_root
                            / batch_runner.model_output_tag(model_name)
                            / basin_name
                            / "train"
                            / "generated_config.yaml"
                        )
                    ).replace("\\", "/"),
                    "error": str(exc),
                }
            all_results.append(result)
            write_summary_csv(all_results, summary_path)
            print(format_result_message(result), flush=True)

    total_elapsed = time.time() - total_started
    print("\n" + "=" * 100, flush=True)
    print(
        "Traditional baseline finished, total time: {:.1f} min ({:.2f} h)".format(
            total_elapsed / 60.0,
            total_elapsed / 3600.0,
        ),
        flush=True,
    )
    print("Summary CSV: {}".format(summary_path), flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
