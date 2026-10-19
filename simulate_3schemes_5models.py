"""Batch simulation runner for 3 schemes x 5 hydrological models.

This script writes a single script-level summary CSV and delegates core
simulation to `simulate.py`.

Key behavior:
- Simulation period is controlled by `simulation.start_date` /
  `simulation.end_date` in the model config.
- No case-level/basin-level time override is applied in this script.
"""

import argparse
import copy
import time
import traceback
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulate import simulate
from utils.seed import kge, nse, rmse, bias


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
    parser = argparse.ArgumentParser(description="Simulate 3 schemes x 5 models")
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
    parser.add_argument(
        "--summary-file",
        type=str,
        default="results/three_schemes/simulate_summary.csv",
        help="Single summary CSV path",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/three_schemes",
        help="Root directory containing per-run training outputs",
    )
    return parser.parse_args()


def _normalize_model_name(name):
    return str(name).strip().lower().replace("-", "").replace("_", "")


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
            "description": "Single-basin simulation in Yangtze headwaters",
            "basins": [make_basin("js", role="both")],
            "sim_basin": "js",
        },
        {
            "name": "hhy_only",
            "description": "Single-basin simulation in Yellow headwaters",
            "basins": [make_basin("hhy", role="both")],
            "sim_basin": "hhy",
        },
    ],
    "scheme2_cross": [
        {
            "name": "train_js_valid_hhy",
            "description": "Use weights trained in js, simulate in hhy",
            "basins": [make_basin("js", role="train"), make_basin("hhy", role="valid")],
            "sim_basin": "hhy",
        },
        {
            "name": "train_hhy_valid_js",
            "description": "Use weights trained in hhy, simulate in js",
            "basins": [make_basin("hhy", role="train"), make_basin("js", role="valid")],
            "sim_basin": "js",
        },
    ],
    "scheme3_joint": [
        {
            "name": "joint_js_hhy_valid_js",
            "description": "Use joint-trained weights, simulate in js",
            "basins": [make_basin("js", role="both", weight=1.0), make_basin("hhy", role="both", weight=1.0)],
            "train_case_name": "joint_js_hhy",
            "sim_basin": "js",
        },
        {
            "name": "joint_js_hhy_valid_hhy",
            "description": "Use joint-trained weights, simulate in hhy",
            "basins": [make_basin("js", role="both", weight=1.0), make_basin("hhy", role="both", weight=1.0)],
            "train_case_name": "joint_js_hhy",
            "sim_basin": "hhy",
        },
    ],
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _get_case_basin_config(case, basin_name):
    basins = case.get("basins", [])
    return next((b for b in basins if b.get("name") == basin_name), basins[0] if basins else {})


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def _yearly_metric_mean(sim_arr, obs_arr, year_index, metric_fn, years=None, base_mask=None):
    """Compute metric year-by-year and return mean across selected years."""
    if base_mask is None:
        base_mask = np.ones(len(year_index), dtype=bool)

    years_filter = None
    if years:
        years_filter = {int(y) for y in years}

    candidate_years = np.unique(year_index[base_mask])
    metric_vals = []
    for yr in sorted(candidate_years.tolist()):
        if years_filter is not None and int(yr) not in years_filter:
            continue
        yr_mask = (year_index == yr) & base_mask
        if not np.any(yr_mask):
            continue
        val = metric_fn(sim_arr[yr_mask], obs_arr[yr_mask])
        if np.isfinite(val):
            metric_vals.append(float(val))

    if not metric_vals:
        return np.nan
    return float(np.mean(metric_vals))


def _compute_station_metrics(sim_arr, obs_arr, dates, train_years, valid_years, test_years, warmup_days=0):
    year_index = pd.DatetimeIndex(dates).year.to_numpy()
    n_times = len(dates)
    warmup_n = int(warmup_days or 0)
    warmup_n = max(0, min(warmup_n, n_times))

    # "all" metrics default to evaluation period excluding warmup section.
    all_mask = np.ones(n_times, dtype=bool)
    if warmup_n > 0:
        all_mask[:warmup_n] = False

    return {
        # All metrics are computed per year first, then averaged across years.
        "KGE_all": _yearly_metric_mean(sim_arr, obs_arr, year_index, metric_fn=kge, base_mask=all_mask),
        "NSE_all": _yearly_metric_mean(sim_arr, obs_arr, year_index, metric_fn=nse, base_mask=all_mask),
        "RMSE_all": _yearly_metric_mean(sim_arr, obs_arr, year_index, metric_fn=rmse, base_mask=all_mask),
        "Bias_all": _yearly_metric_mean(sim_arr, obs_arr, year_index, metric_fn=bias, base_mask=all_mask),
        "KGE_train": _yearly_metric_mean(
            sim_arr, obs_arr, year_index, metric_fn=kge, years=train_years, base_mask=all_mask
        ),
        "KGE_valid": _yearly_metric_mean(
            sim_arr, obs_arr, year_index, metric_fn=kge, years=valid_years, base_mask=all_mask
        ),
        "KGE_test": _yearly_metric_mean(
            sim_arr, obs_arr, year_index, metric_fn=kge, years=test_years, base_mask=all_mask
        ),
    }


def _evaluate_case_metrics(sim_basin, basin_cfg, reach_flow, sim_output_dir):
    """Evaluate per-station metrics based on saved simulation time series and observations."""
    data_root = Path("data") / sim_basin
    model_attr_path = data_root / "model_attr" / "model_attr.json"
    if not model_attr_path.exists():
        return []

    with open(model_attr_path, "r", encoding="utf-8") as f:
        model_attr = json.load(f)

    gauges_cfg = model_attr.get("units", {}).get("gauges", {})
    obs_file = gauges_cfg.get("obs_file")
    if not obs_file:
        return []

    obs_path = Path(obs_file)
    if not obs_path.exists():
        return []

    series_path = Path(sim_output_dir) / "reach_flow_series.csv"
    if not series_path.exists():
        return []
    dates = pd.to_datetime(pd.read_csv(series_path, usecols=["date"])["date"])

    obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True).reindex(dates)
    selected_gauges = basin_cfg.get("selected_gauges", []) or []
    selected_set = {str(g).lower() for g in selected_gauges}

    time_cfg = basin_cfg.get("time", {})
    train_years = time_cfg.get("train_years", [])
    valid_years = time_cfg.get("valid_years", [])
    test_years = time_cfg.get("test_years", [])
    warmup_days = int(time_cfg.get("warmup_days", DEFAULT_TIME.get("warmup_days", 365)) or 0)

    station_metrics = []
    for unit_id, gauge_name in gauges_cfg.items():
        if unit_id in {"obs_file", "comment"}:
            continue
        gauge_name = str(gauge_name)
        if selected_set and gauge_name.lower() not in selected_set:
            continue
        if gauge_name not in obs_df.columns:
            continue

        reach_id = "r{}".format(unit_id)
        if reach_id not in reach_flow:
            continue

        sim_arr = np.asarray(reach_flow[reach_id], dtype=float)
        obs_arr = np.asarray(obs_df[gauge_name].values, dtype=float)

        m = _compute_station_metrics(
            sim_arr,
            obs_arr,
            dates,
            train_years,
            valid_years,
            test_years,
            warmup_days=warmup_days,
        )
        m["reach_id"] = reach_id
        m["gauge_name"] = gauge_name
        station_metrics.append(m)

    return station_metrics


def resolve_default_period_from_default_time():
    """Resolve fallback simulation period from DEFAULT_TIME."""
    years = sorted(
        set(
            DEFAULT_TIME.get("train_years", [])
            + DEFAULT_TIME.get("valid_years", [])
            + DEFAULT_TIME.get("test_years", [])
        )
    )
    if not years:
        raise ValueError("DEFAULT_TIME has no years to build simulation period.")
    start_date = "{}-01-01".format(min(years) - 1)
    end_date = "{}-12-31".format(max(years))
    return start_date, end_date


def run_single_case_simulate(base_config_file, model_name, scheme_name, case, results_root="results/three_schemes"):
    """Run one model in one scheme-case for simulation."""
    sim_basin = case.get("sim_basin", case["basins"][0]["name"])
    train_case_name = case.get("train_case_name", case["name"])

    model_root_dir = Path(results_root) / scheme_name / train_case_name / model_name.lower()
    trained_output_dir = model_root_dir / "train"
    if not trained_output_dir.exists():
        return {
            "scheme": scheme_name,
            "case": case["name"],
            "case_desc": case["description"],
            "model": model_name,
            "sim_basin": sim_basin,
            "status": "failed",
            "error": "Trained model directory not found",
            "time_seconds": 0.0,
            "output_dir": "",
            "generated_config": "",
            "num_stations": 0,
            "avg_kge": np.nan,
            "avg_nse": np.nan,
            "avg_rmse": np.nan,
            "avg_bias": np.nan,
            "avg_kge_train": np.nan,
            "avg_kge_valid": np.nan,
            "avg_kge_test": np.nan,
        }

    trained_config_path = trained_output_dir / "generated_config.yaml"
    if trained_config_path.exists():
        config = load_yaml(trained_config_path)
        print(f"  [Config] Using trained generated config: {trained_config_path}")
    else:
        config = load_yaml(base_config_file)
        print(f"  [Config] Using base config: {base_config_file}")

    sim_output_dir = model_root_dir / "simulate" / sim_basin
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    # Override experiment setup. Simulation period stays in simulation.start/end.
    config["basins"] = copy.deepcopy(case["basins"])
    config["training"]["output_dir"] = str(trained_output_dir).replace("\\", "/")

    sim_cfg = copy.deepcopy(config.get("simulation", {}))
    sim_cfg["output_dir"] = str(sim_output_dir).replace("\\", "/")
    sim_cfg["use_gpu"] = False
    sim_cfg["save_physical_processes"] = sim_cfg.get("save_physical_processes", True)
    # Unified default period: if start/end missing, use DEFAULT_TIME-derived period.
    default_start, default_end = resolve_default_period_from_default_time()
    if not sim_cfg.get("start_date"):
        sim_cfg["start_date"] = default_start
    if not sim_cfg.get("end_date"):
        sim_cfg["end_date"] = default_end
    config["simulation"] = sim_cfg

    generated_config_path = sim_output_dir / "generated_config.yaml"
    save_yaml(generated_config_path, config)

    start_time = time.time()
    error_msg = ""
    all_metrics = []

    try:
        reach_flow, _, _ = simulate(config, basin_name=sim_basin)
        basin_cfg = _get_case_basin_config(case, sim_basin)
        all_metrics = _evaluate_case_metrics(
            sim_basin=sim_basin,
            basin_cfg=basin_cfg,
            reach_flow=reach_flow,
            sim_output_dir=sim_output_dir,
        )
        status = "success"
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        traceback.print_exc()

    elapsed = time.time() - start_time

    # Kept for compatibility with old summary schema.
    num_stations = len(all_metrics) if all_metrics else 0
    avg_kge = np.nan
    avg_nse = np.nan
    avg_rmse = np.nan
    avg_bias = np.nan
    avg_kge_train = np.nan
    avg_kge_valid = np.nan
    avg_kge_test = np.nan

    if status == "success" and all_metrics:
        avg_kge = _nanmean([m.get("KGE_all", np.nan) for m in all_metrics])
        avg_nse = _nanmean([m.get("NSE_all", np.nan) for m in all_metrics])
        avg_rmse = _nanmean([m.get("RMSE_all", np.nan) for m in all_metrics])
        avg_bias = _nanmean([m.get("Bias_all", np.nan) for m in all_metrics])
        avg_kge_train = _nanmean([m.get("KGE_train", np.nan) for m in all_metrics])
        avg_kge_valid = _nanmean([m.get("KGE_valid", np.nan) for m in all_metrics])
        avg_kge_test = _nanmean([m.get("KGE_test", np.nan) for m in all_metrics])

    return {
        "scheme": scheme_name,
        "case": case["name"],
        "case_desc": case["description"],
        "model": model_name,
        "sim_basin": sim_basin,
        "status": status,
        "num_stations": num_stations,
        "avg_kge": avg_kge,
        "avg_nse": avg_nse,
        "avg_rmse": avg_rmse,
        "avg_bias": avg_bias,
        "avg_kge_train": avg_kge_train,
        "avg_kge_valid": avg_kge_valid,
        "avg_kge_test": avg_kge_test,
        "time_seconds": elapsed,
        "output_dir": str(sim_output_dir).replace("\\", "/"),
        "generated_config": str(generated_config_path).replace("\\", "/"),
        "error": error_msg,
    }


def main():
    args = parse_args()
    selected_models = select_model_configs(args.models)
    selected_schemes = select_schemes(args.schemes, args.cases)

    print("=" * 100)
    print("3 schemes x 5 models simulation started")
    print("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Models: {}".format(", ".join(m["name"] for m in selected_models)))
    print("Schemes: {}".format(", ".join(selected_schemes.keys())))
    print("=" * 100)

    all_results = []
    total_start = time.time()

    for scheme_name, cases in selected_schemes.items():
        print("\n" + "#" * 100)
        print("# Running {}".format(scheme_name))
        print("#" * 100)

        for case in cases:
            print("\n" + "-" * 100)
            print(
                "Case: {} | {} | sim_basin: {}".format(
                    case["name"],
                    case["description"],
                    case.get("sim_basin", ""),
                )
            )
            print("-" * 100)

            for i, model_info in enumerate(selected_models, 1):
                print(
                    "\n[{} / {}] {} | {} | {}".format(
                        i, len(selected_models), model_info["name"], scheme_name, case["name"]
                    )
                )

                result = run_single_case_simulate(
                    base_config_file=model_info["config_file"],
                    model_name=model_info["name"],
                    scheme_name=scheme_name,
                    case=case,
                    results_root=args.results_root,
                )
                all_results.append(result)

                if result["status"] == "success":
                    print(
                        "  Success | avg_kge(all)={:.4f} | train={:.4f} | valid={:.4f} | test={:.4f} | "
                        "avg_nse={:.4f} | avg_rmse={:.4f} | avg_bias={:.4f} | stations={} | time={:.1f} min".format(
                            result["avg_kge"],
                            result["avg_kge_train"],
                            result["avg_kge_valid"],
                            result["avg_kge_test"],
                            result["avg_nse"],
                            result["avg_rmse"],
                            result["avg_bias"],
                            result["num_stations"],
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
    print("All simulations finished, total time: {:.1f} min ({:.2f} h)".format(total_elapsed / 60, total_elapsed / 3600))
    print("Summary CSV: {}".format(summary_path))
    print("=" * 100)


if __name__ == "__main__":
    main()
