"""Batch training runner for 3 schemes x 5 hydrological models.

This script now follows the refactored train.py entry:
- dl / scratch
- dl / continue
- traditional / manual
- traditional / differential_evolution

Each task writes its own generated config under the run output directory.
The script-level summary CSV is still written once at the end.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import train as train_module


MODEL_CONFIGS = [
    {"name": "XAJ", "config_file": "configs/config_xaj_cold.yaml"},
    {"name": "GR4J", "config_file": "configs/config_gr4j_cold.yaml"},
    {"name": "SAC-SMA", "config_file": "configs/config_sacsma_cold.yaml"},
    {"name": "Tank", "config_file": "configs/config_tank_cold.yaml"},
    {"name": "HBV", "config_file": "configs/config_hbv_cold.yaml"},
]


DEFAULT_TIME = {
    "train_years": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
    "valid_years": [2013, 2014, 2015],
    "test_years": [2016, 2017, 2018, 2019],
    "warmup_days": 365,
}


SELECTED_GAUGES = {
    "js": ["zhimenda", "batang", "shigu"],
    "hhy": ["maqu", "tangnaihai"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3 schemes x 5 models")
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
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs for quick testing")
    parser.add_argument(
        "--summary-file",
        type=str,
        default="results/three_schemes/train_summary.csv",
        help="Single summary CSV path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/three_schemes",
        help="Root directory for per-run outputs",
    )
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
        help=(
            "Root directory for dl/continue checkpoints. The script appends "
            "<scheme>/<case>/<model>/train automatically. If omitted, continue-training "
            "uses the current run output directory."
        ),
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


def _normalize_model_name(name):
    return str(name).strip().lower().replace("-", "").replace("_", "")


def normalize_mode_strategy(category: str, strategy: Optional[str]) -> Tuple[str, str]:
    category_norm = str(category).strip().lower()
    strategy_norm = str(strategy).strip().lower() if strategy else ""

    if category_norm == "dl":
        aliases = {
            "": "scratch",
            "from_scratch": "scratch",
            "from-scratch": "scratch",
            "resume": "continue",
            "continued": "continue",
        }
        resolved = aliases.get(strategy_norm, strategy_norm)
        if resolved not in {"scratch", "continue"}:
            raise ValueError("DL strategy must be one of scratch/continue")
        return category_norm, resolved

    if category_norm != "traditional":
        raise ValueError("param category must be 'dl' or 'traditional'")

    aliases = {
        "": "differential_evolution",
        "de": "differential_evolution",
        "differential-evolution": "differential_evolution",
        "manual_calibration": "manual",
        "manual-calibration": "manual",
    }
    resolved = aliases.get(strategy_norm, strategy_norm)
    if resolved not in {"manual", "differential_evolution"}:
        raise ValueError("Traditional strategy must be one of manual/differential_evolution")
    return category_norm, resolved


def model_output_tag(model_name: str) -> str:
    return str(model_name).strip().lower()


def select_model_configs(requested_models=None):
    if not requested_models:
        return copy.deepcopy(MODEL_CONFIGS)

    alias_to_name = {
        _normalize_model_name("XAJ"): "XAJ",
        _normalize_model_name("GR4J"): "GR4J",
        _normalize_model_name("SAC-SMA"): "SAC-SMA",
        _normalize_model_name("SACSMA"): "SAC-SMA",
        _normalize_model_name("Tank"): "Tank",
        _normalize_model_name("HBV"): "HBV",
    }

    requested = set()
    for item in requested_models:
        key = _normalize_model_name(item)
        if key not in alias_to_name:
            raise ValueError("Unknown model: {}".format(item))
        requested.add(alias_to_name[key])

    return [m for m in MODEL_CONFIGS if m["name"] in requested]


def select_schemes(requested_schemes=None, requested_cases=None):
    scheme_filter = set(requested_schemes or [])
    case_filter = set(requested_cases or [])
    selected = {}

    for scheme_name, cases in SCHEMES.items():
        if scheme_filter and scheme_name not in scheme_filter:
            continue

        chosen_cases = []
        for case in cases:
            if case_filter and case["name"] not in case_filter:
                continue
            chosen_cases.append(copy.deepcopy(case))

        if chosen_cases:
            selected[scheme_name] = chosen_cases

    if not selected:
        raise ValueError("No scheme/case selected. Check --schemes and --cases")

    return selected


def patch_observation_loader_for_selected_gauges():
    """Patch train loader to filter gauges by basin_config.selected_gauges."""
    if getattr(train_module, "_selected_gauge_patch_applied", False):
        return

    original_loader = train_module.load_observations_for_basin

    def _wrapped_loader(model_attr, dates, basin_config, data_type="train"):
        observations = original_loader(model_attr, dates, basin_config, data_type)
        selected = basin_config.get("selected_gauges", [])

        if not observations or not selected:
            return observations

        selected_set = {str(name).lower() for name in selected}
        gauges = model_attr.get("units", {}).get("gauges", {})

        allowed_reaches = set()
        for unit_id, gauge_name in gauges.items():
            if unit_id in {"obs_file", "comment"}:
                continue
            if str(gauge_name).lower() in selected_set:
                allowed_reaches.add("r{}".format(unit_id))

        filtered = {k: v for k, v in observations.items() if k in allowed_reaches}
        print(
            "  [GaugeFilter] Basin={} {}: {} -> {} gauges | selected={}".format(
                basin_config.get("name"),
                data_type,
                len(observations),
                len(filtered),
                selected,
            )
        )

        return filtered if filtered else None

    train_module.load_observations_for_basin = _wrapped_loader
    train_module._selected_gauge_patch_applied = True


def make_basin(basin_name, role="both", weight=1.0):
    descriptions = {
        "js": "Yangtze headwaters basin",
        "hhy": "Yellow headwaters basin",
    }
    return {
        "name": basin_name,
        "enabled": True,
        "description": descriptions.get(basin_name, basin_name),
        "weight": weight,
        "role": role,
        "selected_gauges": copy.deepcopy(SELECTED_GAUGES.get(basin_name, [])),
        "time": copy.deepcopy(DEFAULT_TIME),
    }


SCHEMES = {
    "scheme1_separate": [
        {
            "name": "js_only",
            "description": "Single-basin training/validation in Yangtze headwaters",
            "basins": [make_basin("js", role="both")],
        },
        {
            "name": "hhy_only",
            "description": "Single-basin training/validation in Yellow headwaters",
            "basins": [make_basin("hhy", role="both")],
        },
    ],
    "scheme2_cross": [
        {
            "name": "train_js_valid_hhy",
            "description": "Train on Yangtze headwaters and validate on Yellow headwaters",
            "basins": [make_basin("js", role="train"), make_basin("hhy", role="valid")],
        },
        {
            "name": "train_hhy_valid_js",
            "description": "Train on Yellow headwaters and validate on Yangtze headwaters",
            "basins": [make_basin("hhy", role="train"), make_basin("js", role="valid")],
        },
    ],
    "scheme3_joint": [
        {
            "name": "joint_js_hhy",
            "description": "Joint training/validation using both basins",
            "basins": [make_basin("js", role="both", weight=1.0), make_basin("hhy", role="both", weight=1.0)],
        },
    ],
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def resolve_checkpoint_dir(
    checkpoint_root: Optional[str],
    scheme_name: str,
    case_name: str,
    model_name: str,
    run_output_dir: Path,
) -> str:
    if checkpoint_root:
        return str(
            Path(checkpoint_root) / scheme_name / case_name / model_output_tag(model_name) / "train"
        ).replace("\\", "/")
    return str(run_output_dir).replace("\\", "/")


def apply_param_mode_overrides(
    config: dict,
    args,
    scheme_name: str,
    case_name: str,
    model_name: str,
    run_output_dir: Path,
) -> Dict[str, Optional[str]]:
    category, strategy = normalize_mode_strategy(args.param_category, args.param_strategy)
    checkpoint_dir = None

    components = config.setdefault("model", {}).setdefault("components", {})
    active_components = train_module.get_active_component_names(config)

    for component_name in active_components:
        component_cfg = components.setdefault(component_name, {})
        if category == "dl":
            component_mode = {
                "category": "dl",
                "strategy": strategy,
            }
            if strategy == "continue":
                checkpoint_dir = resolve_checkpoint_dir(
                    args.checkpoint_root,
                    scheme_name=scheme_name,
                    case_name=case_name,
                    model_name=model_name,
                    run_output_dir=run_output_dir,
                )
                component_mode["checkpoint_dir"] = checkpoint_dir
            component_cfg["param_mode"] = component_mode
        else:
            component_cfg["param_mode"] = {
                "category": "traditional",
                "strategy": strategy,
            }

    training_cfg = config.setdefault("training", {})
    training_cfg.pop("resume", None)

    if category == "traditional":
        traditional_cfg = copy.deepcopy(training_cfg.get("traditional", {}))
        overrides = {
            "maxiter": args.de_maxiter,
            "popsize": args.de_popsize,
            "mutation": args.de_mutation,
            "recombination": args.de_recombination,
            "tol": args.de_tol,
        }
        for key, value in overrides.items():
            if value is not None:
                traditional_cfg[key] = value
        if args.write_to_data_params:
            traditional_cfg["write_to_data_params"] = True
        training_cfg["traditional"] = traditional_cfg
    else:
        training_cfg.pop("traditional", None)

    return {
        "param_category": category,
        "param_strategy": strategy,
        "checkpoint_dir": checkpoint_dir,
    }


def extract_run_metrics(run_output_dir: Path, mode_info: dict, history: Optional[dict]) -> Tuple[float, float, int]:
    if mode_info["param_category"] == "dl":
        history = history or {}
        valid_hist = history.get("avg_valid_kge", [])
        train_hist = history.get("avg_train_kge", [])
        best_valid_kge = max(valid_hist) if valid_hist else np.nan
        final_train_kge = train_hist[-1] if train_hist else np.nan
        total_epochs = len(history.get("epoch", []))
        return best_valid_kge, final_train_kge, total_epochs

    if mode_info["param_strategy"] == "differential_evolution":
        summary_file = run_output_dir / "calibration_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best_valid_kge = float(summary.get("valid", {}).get("avg_kge", np.nan))
            final_train_kge = float(summary.get("train", {}).get("avg_kge", np.nan))
            total_epochs = int(summary.get("optimization_result", {}).get("iterations_completed", 0))
            return best_valid_kge, final_train_kge, total_epochs

    return np.nan, np.nan, 0


def run_single_case(base_config_file, model_name, scheme_name, case, args):
    """Run one model in one scheme-case."""
    config = load_yaml(base_config_file)

    run_output_dir = Path(args.output_root) / scheme_name / case["name"] / model_output_tag(model_name) / "train"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    config["basins"] = copy.deepcopy(case["basins"])
    config["training"]["output_dir"] = str(run_output_dir).replace("\\", "/")
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    mode_info = apply_param_mode_overrides(
        config=config,
        args=args,
        scheme_name=scheme_name,
        case_name=case["name"],
        model_name=model_name,
        run_output_dir=run_output_dir,
    )

    generated_config_path = run_output_dir / "generated_config.yaml"
    save_yaml(generated_config_path, config)

    start_time = time.time()
    history = None
    error_msg = ""

    try:
        _, history = train_module.train(config, config_path=str(generated_config_path))
        status = "success"
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        traceback.print_exc()

    elapsed = time.time() - start_time

    if status == "success":
        best_valid_kge, final_train_kge, total_epochs = extract_run_metrics(
            run_output_dir=run_output_dir,
            mode_info=mode_info,
            history=history,
        )
    else:
        best_valid_kge = np.nan
        final_train_kge = np.nan
        total_epochs = 0

    return {
        "scheme": scheme_name,
        "case": case["name"],
        "case_desc": case["description"],
        "model": model_name,
        "param_category": mode_info["param_category"],
        "param_strategy": mode_info["param_strategy"],
        "checkpoint_dir": mode_info["checkpoint_dir"] or "",
        "status": status,
        "best_valid_kge": best_valid_kge,
        "final_train_kge": final_train_kge,
        "total_epochs": total_epochs,
        "time_seconds": elapsed,
        "output_dir": str(run_output_dir).replace("\\", "/"),
        "generated_config": str(generated_config_path).replace("\\", "/"),
        "error": error_msg,
    }


def main():
    args = parse_args()
    patch_observation_loader_for_selected_gauges()
    selected_models = select_model_configs(args.models)
    selected_schemes = select_schemes(args.schemes, args.cases)
    mode_category, mode_strategy = normalize_mode_strategy(args.param_category, args.param_strategy)

    print("=" * 100)
    print("3 schemes x 5 models training started")
    print("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Models: {}".format(", ".join(m["name"] for m in selected_models)))
    print("Schemes: {}".format(", ".join(selected_schemes.keys())))
    print("Param mode: {}/{}".format(mode_category, mode_strategy))
    if mode_category == "dl" and mode_strategy == "continue":
        print("Checkpoint root: {}".format(args.checkpoint_root or "<current run output dir>"))
    if args.epochs is not None:
        print("epochs override: {}".format(args.epochs))
    print("=" * 100)

    all_results = []
    total_start = time.time()

    for scheme_name, cases in selected_schemes.items():
        print("\n" + "#" * 100)
        print("# Running {}".format(scheme_name))
        print("#" * 100)

        for case in cases:
            print("\n" + "-" * 100)
            print("Case: {} | {}".format(case["name"], case["description"]))
            print("-" * 100)

            for i, model_info in enumerate(selected_models, 1):
                print(
                    "\n[{} / {}] {} | {} | {} | {}/{}".format(
                        i,
                        len(selected_models),
                        model_info["name"],
                        scheme_name,
                        case["name"],
                        mode_category,
                        mode_strategy,
                    )
                )

                result = run_single_case(
                    base_config_file=model_info["config_file"],
                    model_name=model_info["name"],
                    scheme_name=scheme_name,
                    case=case,
                    args=args,
                )
                all_results.append(result)

                if result["status"] == "success":
                    print(
                        "  Success | best_valid_kge={:.4f} | final_train_kge={:.4f} | epochs={} | time={:.1f} min".format(
                            result["best_valid_kge"],
                            result["final_train_kge"],
                            result["total_epochs"],
                            result["time_seconds"] / 60,
                        )
                    )
                else:
                    print("  Failed | error={}".format(result["error"]))

    total_elapsed = time.time() - total_start

    summary_df = pd.DataFrame(all_results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["scheme", "case", "model"], ascending=[True, True, True])

    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("All training finished, total time: {:.1f} min ({:.2f} h)".format(total_elapsed / 60, total_elapsed / 3600))
    print("Summary CSV: {}".format(summary_path))
    print("=" * 100)


if __name__ == "__main__":
    main()
