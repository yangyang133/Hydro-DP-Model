from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module
from utils.seed import bias, kge, nse, rmse, set_global_seed


@dataclass(frozen=True)
class ParameterSpec:
    group: str
    name: str
    lower: float
    upper: float
    initial: float


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional differential-evolution calibration")
    parser.add_argument("--config", type=str, required=True, help="Runtime YAML config path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--maxiter", type=int, default=None, help="Maximum DE generations")
    parser.add_argument("--popsize", type=int, default=None, help="Population size")
    parser.add_argument("--mutation", type=float, default=None, help="DE mutation factor")
    parser.add_argument("--recombination", type=float, default=None, help="DE crossover rate")
    parser.add_argument("--tol", type=float, default=None, help="Early-stop tolerance on population std")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--write-to-data-params",
        action="store_true",
        help="Also copy calibrated parameter CSVs to data/<basin>/params",
    )
    return parser.parse_args()


def load_runtime_config(config_path: str, args) -> dict:
    config = train_module.load_config(config_path)
    config["training"]["output_dir"] = str(Path(args.output_dir)).replace("\\", "/")
    traditional_cfg = config.setdefault("training", {}).setdefault("traditional", {})
    overrides = {
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "mutation": args.mutation,
        "recombination": args.recombination,
        "tol": args.tol,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            traditional_cfg[key] = value
    if args.write_to_data_params:
        traditional_cfg["write_to_data_params"] = True
    return config


def select_device(config: dict) -> torch.device:
    use_gpu = bool(config.get("training", {}).get("use_gpu", False))
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def build_basin_context(config: dict):
    enabled_basins = train_module.get_enabled_basins(config)
    if not enabled_basins:
        raise ValueError("No enabled basins found in configuration!")

    basins = {}
    for basin_cfg in enabled_basins:
        basin_data = train_module.load_basin_data(basin_cfg, config)
        basins[basin_data.name] = basin_data

    train_basins = {
        name: data for name, data in basins.items()
        if getattr(data, "role", "both") in {"both", "train"}
    }
    valid_basins = {
        name: data for name, data in basins.items()
        if getattr(data, "role", "both") in {"both", "valid"}
    }

    if not train_basins:
        raise ValueError("No basins available for training (role in {'both', 'train'})")
    if not valid_basins:
        raise ValueError("No basins available for validation (role in {'both', 'valid'})")

    basin_gauge_names = {}
    for basin_name, basin_data in basins.items():
        gauges = basin_data.model_attr.get("units", {}).get("gauges", {})
        names = {}
        for key, gauge_name in gauges.items():
            if key in {"obs_file", "comment"}:
                continue
            names[f"r{key}"] = gauge_name
        basin_gauge_names[basin_name] = names

    first_basin = next(iter(basins.values()))
    warmup = first_basin.model_attr.get("units", {}).get(
        "warmup_days",
        config.get("time", {}).get("warmup_days", 365),
    )
    kge_mode = config.get("training", {}).get("kge_mode", "series")
    return basins, train_basins, valid_basins, basin_gauge_names, int(warmup), kge_mode


def build_parameter_specs(example_basin) -> List[ParameterSpec]:
    param_info = example_basin.model.get_param_info()
    specs: List[ParameterSpec] = []

    sub_info = param_info["subbasins"]["params_per_subbasin"]["static"]
    river_info = param_info["rivers"]["params_per_reach"]["static"]

    for name, bounds, initial in zip(sub_info["names"], sub_info["bounds"], sub_info["init"]):
        specs.append(
            ParameterSpec(
                group="subbasin",
                name=str(name),
                lower=float(bounds[0]),
                upper=float(bounds[1]),
                initial=float(initial),
            )
        )

    for name, bounds, initial in zip(river_info["names"], river_info["bounds"], river_info["init"]):
        specs.append(
            ParameterSpec(
                group="river",
                name=str(name),
                lower=float(bounds[0]),
                upper=float(bounds[1]),
                initial=float(initial),
            )
        )

    if not specs:
        raise ValueError("No static parameters found for differential evolution calibration.")
    return specs


def vector_to_params_dict(basin_data, specs: List[ParameterSpec], vector: np.ndarray) -> dict:
    grouped_values = {(spec.group, spec.name): float(value) for spec, value in zip(specs, vector)}
    params = {"subbasins": {}, "rivers": {}}

    sub_specs = [spec for spec in specs if spec.group == "subbasin"]
    river_specs = [spec for spec in specs if spec.group == "river"]

    for sub_id in basin_data.model.subbasin_ids:
        params["subbasins"][sub_id] = {"static": {}, "dynamic": {}}
        for spec in sub_specs:
            params["subbasins"][sub_id]["static"][spec.name] = grouped_values[(spec.group, spec.name)]

    for reach_id in basin_data.model.reaches.keys():
        params["rivers"][reach_id] = {"static": {}, "dynamic": {}}
        for spec in river_specs:
            params["rivers"][reach_id]["static"][spec.name] = grouped_values[(spec.group, spec.name)]

    return params


def yearly_metric_mean(sim_arr, obs_arr, dates, metric_fn) -> float:
    year_index = pd.DatetimeIndex(dates).year.to_numpy()
    values = []
    for year in sorted(np.unique(year_index).tolist()):
        mask = year_index == year
        if not np.any(mask):
            continue
        metric_val = metric_fn(sim_arr[mask], obs_arr[mask])
        if np.isfinite(metric_val):
            values.append(float(metric_val))
    return float(np.mean(values)) if values else float("nan")


def collect_station_rows(
    basin_name: str,
    period_name: str,
    outputs: dict,
    obs: dict,
    dates,
    warmup: int,
    gauge_names: dict,
) -> List[dict]:
    rows = []
    river_flows = outputs.get("river_flows", {})
    eval_dates = pd.DatetimeIndex(dates[warmup:])

    for reach_id, obs_tuple in (obs or {}).items():
        if reach_id not in river_flows:
            continue
        obs_values, weight = obs_tuple
        sim_arr = np.asarray(river_flows[reach_id].detach().cpu().numpy(), dtype=float)[warmup:]
        obs_arr = np.asarray(obs_values, dtype=float)[warmup:]
        valid_mask = np.isfinite(sim_arr) & np.isfinite(obs_arr)

        rows.append(
            {
                "basin": basin_name,
                "period": period_name,
                "reach_id": reach_id,
                "gauge_name": gauge_names.get(reach_id, reach_id),
                "weight": float(weight),
                "n_total": int(obs_arr.size),
                "n_valid": int(valid_mask.sum()),
                "KGE": float(kge(sim_arr, obs_arr)),
                "NSE": float(nse(sim_arr, obs_arr)),
                "RMSE": float(rmse(sim_arr, obs_arr)),
                "Bias": float(bias(sim_arr, obs_arr)),
                "KGE_yearly": float(yearly_metric_mean(sim_arr, obs_arr, eval_dates, kge)),
            }
        )

    return rows


def summarize_station_rows(station_rows: List[dict], objective: float, basin_kge: Dict[str, float]) -> List[dict]:
    rows = []
    if not station_rows:
        rows.append(
            {
                "period": "__all__",
                "basin": "__all__",
                "objective": float(objective),
                "avg_kge_objective": float(np.nan),
                "avg_kge_series": float(np.nan),
                "avg_nse": float(np.nan),
                "avg_rmse": float(np.nan),
                "avg_bias": float(np.nan),
                "n_stations": 0,
            }
        )
        return rows

    df = pd.DataFrame(station_rows)
    all_avg_kge = float(np.nanmean(list(basin_kge.values()))) if basin_kge else float("nan")
    rows.append(
        {
            "period": df["period"].iloc[0],
            "basin": "__all__",
            "objective": float(objective),
            "avg_kge_objective": all_avg_kge,
            "avg_kge_series": float(df["KGE"].mean()),
            "avg_nse": float(df["NSE"].mean()),
            "avg_rmse": float(df["RMSE"].mean()),
            "avg_bias": float(df["Bias"].mean()),
            "n_stations": int(len(df)),
        }
    )

    for basin_name, basin_df in df.groupby("basin"):
        rows.append(
            {
                "period": basin_df["period"].iloc[0],
                "basin": basin_name,
                "objective": float(objective),
                "avg_kge_objective": float(basin_kge.get(basin_name, np.nan)),
                "avg_kge_series": float(basin_df["KGE"].mean()),
                "avg_nse": float(basin_df["NSE"].mean()),
                "avg_rmse": float(basin_df["RMSE"].mean()),
                "avg_bias": float(basin_df["Bias"].mean()),
                "n_stations": int(len(basin_df)),
            }
        )
    return rows


def evaluate_vector(
    vector: np.ndarray,
    split_basins: Dict[str, object],
    period_name: str,
    specs: List[ParameterSpec],
    basin_gauge_names: dict,
    warmup: int,
    kge_mode: str,
    device: torch.device,
) -> dict:
    total_objective = 0.0
    basin_kge = {}
    station_rows = []

    for basin_name, basin_data in split_basins.items():
        params = vector_to_params_dict(basin_data, specs, vector)
        if period_name == "train":
            forcing = {"P": basin_data.P_train, "E": basin_data.E_train}
            if basin_data.T_train is not None:
                forcing["T"] = basin_data.T_train
            obs = basin_data.obs_train
            dates = basin_data.dates_train
        else:
            forcing = {"P": basin_data.P_valid, "E": basin_data.E_valid}
            if basin_data.T_valid is not None:
                forcing["T"] = basin_data.T_valid
            obs = basin_data.obs_valid
            dates = basin_data.dates_valid

        with torch.no_grad():
            outputs, _ = basin_data.model.forward(forcing, params=params, device=device)
        basin_loss, basin_kge_value, _ = train_module._compute_loss(
            outputs,
            obs,
            warmup,
            device,
            dates=dates,
            kge_mode=kge_mode,
            gauge_names=basin_gauge_names.get(basin_name, {}),
        )

        total_objective += float(basin_loss.detach().cpu().item()) * float(basin_data.weight)
        basin_kge[basin_name] = float(basin_kge_value)
        station_rows.extend(
            collect_station_rows(
                basin_name=basin_name,
                period_name=period_name,
                outputs=outputs,
                obs=obs,
                dates=dates,
                warmup=warmup,
                gauge_names=basin_gauge_names.get(basin_name, {}),
            )
        )

    avg_kge = float(np.nanmean(list(basin_kge.values()))) if basin_kge else float("nan")
    return {
        "objective": float(total_objective),
        "avg_kge": avg_kge,
        "basin_kge": basin_kge,
        "station_rows": station_rows,
        "period_summary": summarize_station_rows(station_rows, total_objective, basin_kge),
    }


def run_differential_evolution(
    objective_fn,
    initial_vector: np.ndarray,
    bounds: List[Tuple[float, float]],
    maxiter: int,
    popsize: int,
    mutation: float,
    recombination: float,
    tol: float,
    rng: np.random.Generator,
):
    dim = len(bounds)
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    popsize = max(int(popsize), 4)

    population = rng.uniform(lower, upper, size=(popsize, dim))
    population[0] = np.clip(np.asarray(initial_vector, dtype=float), lower, upper)

    scores = np.empty(popsize, dtype=float)
    train_kges = np.empty(popsize, dtype=float)
    nfev = 0
    for idx in range(popsize):
        result = objective_fn(population[idx])
        scores[idx] = float(result["objective"])
        train_kges[idx] = float(result["avg_kge"])
        nfev += 1

    history_rows = []

    def append_history(iteration_idx: int):
        best_idx = int(np.argmin(scores))
        history_rows.append(
            {
                "iteration": int(iteration_idx),
                "nfev": int(nfev),
                "best_objective": float(scores[best_idx]),
                "population_mean": float(np.mean(scores)),
                "population_std": float(np.std(scores)),
                "best_train_kge": float(train_kges[best_idx]),
            }
        )

    append_history(0)
    if maxiter <= 0:
        best_idx = int(np.argmin(scores))
        return population[best_idx].copy(), float(scores[best_idx]), history_rows, int(nfev), 0

    for generation in range(1, maxiter + 1):
        for idx in range(popsize):
            candidates = [j for j in range(popsize) if j != idx]
            a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False)
            mutant = population[a_idx] + float(mutation) * (population[b_idx] - population[c_idx])
            mutant = np.clip(mutant, lower, upper)

            crossover_mask = rng.random(dim) < float(recombination)
            if not np.any(crossover_mask):
                crossover_mask[rng.integers(0, dim)] = True
            trial = np.where(crossover_mask, mutant, population[idx])

            result = objective_fn(trial)
            trial_score = float(result["objective"])
            trial_train_kge = float(result["avg_kge"])
            nfev += 1

            if trial_score <= scores[idx]:
                population[idx] = trial
                scores[idx] = trial_score
                train_kges[idx] = trial_train_kge

        append_history(generation)
        if float(np.std(scores)) <= float(tol):
            best_idx = int(np.argmin(scores))
            return population[best_idx].copy(), float(scores[best_idx]), history_rows, int(nfev), generation

    best_idx = int(np.argmin(scores))
    return population[best_idx].copy(), float(scores[best_idx]), history_rows, int(nfev), maxiter


def write_parameter_outputs(
    output_dir: Path,
    basins: Dict[str, object],
    specs: List[ParameterSpec],
    best_vector: np.ndarray,
    write_to_data_params: bool,
):
    params_root = output_dir / "params"
    params_root.mkdir(parents=True, exist_ok=True)
    grouped_values = {(spec.group, spec.name): float(value) for spec, value in zip(specs, best_vector)}

    sub_specs = [spec for spec in specs if spec.group == "subbasin"]
    river_specs = [spec for spec in specs if spec.group == "river"]

    for basin_name, basin_data in basins.items():
        basin_dir = params_root / basin_name
        basin_dir.mkdir(parents=True, exist_ok=True)

        sub_rows = []
        for sub_id in basin_data.model.subbasin_ids:
            unit_id = str(sub_id).replace("sub_", "", 1)
            row = {"unit_id": unit_id}
            for spec in sub_specs:
                row[spec.name] = grouped_values[(spec.group, spec.name)]
            sub_rows.append(row)

        subbasin_model_name = config_component_model(basin_data.model.full_config, "subbasin_runoff", "xaj")
        sub_file = basin_dir / f"{subbasin_model_name}_params.csv"
        pd.DataFrame(sub_rows).to_csv(sub_file, index=False)

        river_rows = []
        for reach_id in basin_data.model.reaches.keys():
            row = {"reach_id": reach_id}
            for spec in river_specs:
                row[spec.name] = grouped_values[(spec.group, spec.name)]
            river_rows.append(row)

        river_model_name = config_component_model(basin_data.model.full_config, "river_routing", "muskingum")
        river_file = basin_dir / f"{river_model_name}_params.csv"
        pd.DataFrame(river_rows).to_csv(river_file, index=False)

        if write_to_data_params:
            target_dir = ROOT / "data" / basin_name / "params"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sub_file, target_dir / sub_file.name)
            shutil.copy2(river_file, target_dir / river_file.name)


def config_component_model(config: dict, component_name: str, default: str) -> str:
    return str(config.get("model", {}).get("components", {}).get(component_name, {}).get("model", default))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_runtime_config(args.config, args)
    traditional_cfg = config.get("training", {}).get("traditional", {})
    maxiter = int(traditional_cfg.get("maxiter", 40))
    popsize = int(traditional_cfg.get("popsize", 8))
    mutation = float(traditional_cfg.get("mutation", 0.8))
    recombination = float(traditional_cfg.get("recombination", 0.7))
    tol = float(traditional_cfg.get("tol", 0.01))
    seed = traditional_cfg.get("seed", config.get("training", {}).get("seed", 42))
    seed = 42 if seed is None else int(seed)
    write_to_data_params = bool(traditional_cfg.get("write_to_data_params", False))

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = select_device(config)

    if train_module.get_mode(config) != "subbasin":
        raise ValueError("Differential evolution calibration is currently supported in subbasin mode only.")

    basins, train_basins, valid_basins, basin_gauge_names, warmup, kge_mode = build_basin_context(config)
    first_basin = next(iter(train_basins.values()))
    specs = build_parameter_specs(first_basin)
    initial_vector = np.asarray([spec.initial for spec in specs], dtype=float)
    bounds = [(spec.lower, spec.upper) for spec in specs]

    evaluation_cache: Dict[Tuple[float, ...], dict] = {}

    def objective_fn(vector: np.ndarray):
        key = tuple(np.round(np.asarray(vector, dtype=float), 10).tolist())
        if key not in evaluation_cache:
            evaluation_cache[key] = evaluate_vector(
                vector=np.asarray(vector, dtype=float),
                split_basins=train_basins,
                period_name="train",
                specs=specs,
                basin_gauge_names=basin_gauge_names,
                warmup=warmup,
                kge_mode=kge_mode,
                device=device,
            )
        return evaluation_cache[key]

    started = time.time()
    best_vector, best_objective, history_rows, nfev, iterations_completed = run_differential_evolution(
        objective_fn=objective_fn,
        initial_vector=initial_vector,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        tol=tol,
        rng=rng,
    )
    elapsed = time.time() - started

    best_train_result = objective_fn(best_vector)
    best_valid_result = evaluate_vector(
        vector=best_vector,
        split_basins=valid_basins,
        period_name="valid",
        specs=specs,
        basin_gauge_names=basin_gauge_names,
        warmup=warmup,
        kge_mode=kge_mode,
        device=device,
    )

    pd.DataFrame(history_rows).to_csv(output_dir / "de_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "group": spec.group,
                "name": spec.name,
                "lower": spec.lower,
                "upper": spec.upper,
                "initial_value": spec.initial,
                "best_value": float(best_value),
            }
            for spec, best_value in zip(specs, best_vector)
        ]
    ).to_csv(output_dir / "parameter_vector.csv", index=False)

    station_df = pd.DataFrame(best_train_result["station_rows"] + best_valid_result["station_rows"])
    station_df.to_csv(output_dir / "station_metrics.csv", index=False)

    period_df = pd.DataFrame(best_train_result["period_summary"] + best_valid_result["period_summary"])
    period_df.to_csv(output_dir / "period_summary.csv", index=False)

    write_parameter_outputs(
        output_dir=output_dir,
        basins=basins,
        specs=specs,
        best_vector=best_vector,
        write_to_data_params=write_to_data_params,
    )

    summary = {
        "config_path": str(args.config),
        "shared_parameterization": True,
        "shared_parameterization_note": (
            "One shared subbasin parameter set and one shared river-routing parameter set "
            "were calibrated and then expanded to all units and reaches."
        ),
        "objective": config.get("training", {}).get("objective", "KGE"),
        "kge_mode": kge_mode,
        "n_parameters": len(specs),
        "de_settings": {
            "maxiter": maxiter,
            "population_size": popsize,
            "mutation": mutation,
            "recombination": recombination,
            "tol": tol,
            "seed": seed,
        },
        "optimization_result": {
            "best_objective": float(best_objective),
            "iterations_completed": int(iterations_completed),
            "function_evaluations": int(nfev),
            "elapsed_seconds": float(elapsed),
        },
        "train": {
            "objective": float(best_train_result["objective"]),
            "avg_kge": float(best_train_result["avg_kge"]),
            "basin_kge": best_train_result["basin_kge"],
        },
        "valid": {
            "objective": float(best_valid_result["objective"]),
            "avg_kge": float(best_valid_result["avg_kge"]),
            "basin_kge": best_valid_result["basin_kge"],
        },
        "initial_vector": [float(x) for x in initial_vector.tolist()],
        "best_vector": [float(x) for x in best_vector.tolist()],
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Traditional calibration mode: differential_evolution")
    print(f"Output directory: {output_dir}")
    print(f"Best train objective: {best_train_result['objective']:.6f}")
    print(f"Best valid objective: {best_valid_result['objective']:.6f}")
    print(f"Train avg KGE: {best_train_result['avg_kge']:.6f}")
    print(f"Valid avg KGE: {best_valid_result['avg_kge']:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
