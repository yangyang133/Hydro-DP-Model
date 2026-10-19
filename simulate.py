"""Base simulation runner.

This script generates hydrological simulation time series and optional
intermediate process/state series. It does not compute metrics, create plots,
or run post-processing analysis.
"""

from pathlib import Path
import argparse
import os
import copy
from typing import Optional
import json

import pandas as pd
import torch
import yaml

from dl.predictor import DynamicParameterPredictor, load_features
from models.hydro_model import HydroModel
from train import build_runtime_config, get_active_component_names, resolve_component_param_modes


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_enabled_basins(config: dict) -> list:
    """Return enabled basin configs."""
    basins = config.get("basins", [])
    return [b for b in basins if b.get("enabled", True)]


def get_basin_data_paths(basin_name: str) -> dict:
    """Build standard data paths for a basin."""
    base_dir = f"data/{basin_name}"
    return {
        "model_attr": f"{base_dir}/model_attr/model_attr.json",
        "dl_input": f"{base_dir}/dl_input",
        "forcing": {
            "precipitation": f"{base_dir}/forcing/P.csv",
            "evaporation": f"{base_dir}/forcing/E.csv",
            "temperature": f"{base_dir}/forcing/T.csv",
        },
    }


def load_forcing_data(
    basin_name: str,
    config: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Load forcing data (P, E, optionally PET by T)."""
    data_paths = get_basin_data_paths(basin_name)
    evap_config = config.get("model", {}).get("evaporation", {})

    p_df = pd.read_csv(data_paths["forcing"]["precipitation"], index_col=0, parse_dates=True)

    use_input_evap = evap_config.get("use_input", True)
    evap_file = data_paths["forcing"]["evaporation"]
    if use_input_evap and os.path.exists(evap_file):
        e_df = pd.read_csv(evap_file, index_col=0, parse_dates=True)
    else:
        temp_file = data_paths["forcing"]["temperature"]
        if not os.path.exists(temp_file):
            raise ValueError("No evaporation input and no temperature file for PET calculation.")
        t_df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
        from models.physics.pet import compute_pet

        pet_method = evap_config.get("pet_method", "hargreaves")
        e_df = compute_pet(t_df, method=pet_method)
        print(f"[Data] Calculated PET using method: {pet_method}")

    if start_date:
        p_df = p_df[p_df.index >= start_date]
        e_df = e_df[e_df.index >= start_date]
    if end_date:
        p_df = p_df[p_df.index <= end_date]
        e_df = e_df[e_df.index <= end_date]

    p = p_df.values
    e = e_df.values
    dates = p_df.index

    print(f"[Data] Forcing loaded: {p.shape[0]} days, {p.shape[1]} subbasins")
    print(f"[Data] Period: {dates[0]} to {dates[-1]}")
    return p, e, dates


def _build_params_dict_for_sim(model: HydroModel, predicted_params: dict) -> dict:
    """Convert predicted parameter vectors to HydroModel parameter dictionary."""
    params_dict = {"subbasins": {}, "rivers": {}}

    param_info = model.get_param_info()
    subbasin_param_names = param_info["subbasins"]["params_per_subbasin"]["static"]["names"]
    river_param_names = param_info["rivers"]["params_per_reach"]["static"]["names"]

    for i, sub_id in enumerate(model.subbasin_ids):
        params_dict["subbasins"][sub_id] = {"static": {}, "dynamic": {}}
        for param_name in subbasin_param_names:
            if param_name in predicted_params:
                params_dict["subbasins"][sub_id]["static"][param_name] = predicted_params[param_name][i]

    def reach_sort_key(reach_id: str):
        suffix = reach_id[1:]
        try:
            return int(suffix), ""
        except ValueError:
            return 0, suffix

    reach_ids_ordered = sorted(
        [rid for rid in model.reaches.keys() if rid.startswith("r")],
        key=reach_sort_key,
    )
    for i, reach_id in enumerate(reach_ids_ordered):
        params_dict["rivers"][reach_id] = {"static": {}, "dynamic": {}}
        for param_name in river_param_names:
            if param_name in predicted_params:
                params_dict["rivers"][reach_id]["static"][param_name] = predicted_params[param_name][i]

    return params_dict


def _build_dl_model_configs(config: dict, resolved_modes: dict) -> dict:
    """Build DynamicParameterPredictor config for active DL components."""
    components = config.get("model", {}).get("components", {})
    model_configs = {}
    for component_name in get_active_component_names(config):
        resolved = resolved_modes[component_name]
        if resolved.category != "dl":
            continue
        component_cfg = components.get(component_name, {})
        model_name = component_cfg.get("model", resolved.model_name)
        model_configs[model_name] = {"param_mode": "trainable", "pretrained_path": ""}
    return model_configs


def _build_simulation_runtime_config(
    config: dict,
    resolved_modes: dict,
    basin_name: str,
    training_output_dir: Path,
) -> dict:
    """Convert new param_mode schema into runtime settings consumed by HydroModel."""
    runtime_config = build_runtime_config(config, resolved_modes)
    sim_config = runtime_config.get("simulation", {})
    components = runtime_config.setdefault("model", {}).setdefault("components", {})

    # Traditional simulation consumes calibrated/manual CSV parameters, so the
    # low-level HydroModel should run these components as fixed parameters.
    uses_traditional = any(mode.category == "traditional" for mode in resolved_modes.values())
    if uses_traditional:
        for component_name, resolved in resolved_modes.items():
            if resolved.category == "traditional":
                components.setdefault(component_name, {})["param_mode"] = "fixed"

    params_dir = sim_config.get("params_dir")
    if not params_dir:
        de_params_dir = training_output_dir / "params" / basin_name
        if de_params_dir.exists():
            params_dir = str(de_params_dir)

    if params_dir:
        runtime_config["_params_dir_override"] = str(params_dir)

    return runtime_config


def _load_trained_networks(predictor: DynamicParameterPredictor, network_dir: Path, device: torch.device, strict: bool):
    """Load DL network weights from a train output directory."""
    loaded_count = 0
    missing = []
    for net_name, network in predictor.networks.items():
        net_file = network_dir / f"{net_name}_net.pt"
        if not net_file.exists():
            missing.append(net_file.name)
            print(f"[Warning] Missing network file: {net_file.name}")
            continue
        try:
            state_dict = torch.load(str(net_file), weights_only=True)
        except TypeError:
            state_dict = torch.load(str(net_file))
        network.load_state_dict(state_dict)
        network.eval()
        network.to(device)
        loaded_count += 1
        print(f"[Loaded] {net_file.name}")

    if strict and missing:
        raise FileNotFoundError(
            "Missing trained network file(s): {} in {}".format(", ".join(missing), network_dir)
        )

    print(f"[Simulate] Networks loaded: {loaded_count}/{len(predictor.networks)}")
    return loaded_count


def _resolve_simulation_period(sim_config: dict, basin_cfg: dict) -> tuple[str, str]:
    """Resolve simulation period from config with clear priority.

    Priority:
    1) simulation.start_date / simulation.end_date
    2) basins[].time.simulate_start_date / simulate_end_date
    3) basins[].time.simulate_years
    4) basins[].time train/valid/test years fallback
    """
    time_cfg = basin_cfg.get("time", {})

    # 1) Global simulation section
    start_date = sim_config.get("start_date")
    end_date = sim_config.get("end_date")

    # 2) Basin-specific simulate start/end
    if not start_date:
        start_date = time_cfg.get("simulate_start_date")
    if not end_date:
        end_date = time_cfg.get("simulate_end_date")

    # 3) Basin-specific simulate years
    sim_years = time_cfg.get("simulate_years", [])
    if (not start_date or not end_date) and sim_years:
        years = sorted(set(int(y) for y in sim_years))
        if not start_date:
            start_date = f"{years[0]}-01-01"
        if not end_date:
            end_date = f"{years[-1]}-12-31"

    # 4) Fallback to train/valid/test years
    if not start_date or not end_date:
        train_years = time_cfg.get("train_years", [])
        valid_years = time_cfg.get("valid_years", [])
        test_years = time_cfg.get("test_years", [])
        all_years = sorted(set(train_years + valid_years + test_years))
        if not all_years:
            raise ValueError(
                "Cannot resolve simulation period. Please set either "
                "simulation.start_date/end_date or basins[].time.simulate_*."
            )
        # Preserve previous default: include one pre-year for warm-up.
        if not start_date:
            start_date = f"{min(all_years) - 1}-01-01"
        if not end_date:
            end_date = f"{max(all_years)}-12-31"

    return str(start_date), str(end_date)


def _save_tensor_series_group(
    tensor_dict: dict,
    dates: pd.DatetimeIndex,
    subbasin_ids: list,
    output_dir: Path,
) -> int:
    """Save {name: tensor[T, n_sub]} groups to CSV files."""
    saved_count = 0
    n_time = len(dates)
    n_sub = len(subbasin_ids)

    for name, value in sorted(tensor_dict.items()):
        if not isinstance(value, torch.Tensor):
            continue
        arr = value.detach().cpu().numpy()
        if arr.ndim != 2 or arr.shape[0] != n_time:
            # Keep base runner strict: only save time-series tensors.
            continue

        if arr.shape[1] == n_sub:
            columns = subbasin_ids
        else:
            columns = [f"col_{i+1}" for i in range(arr.shape[1])]

        df = pd.DataFrame(arr, columns=columns)
        df.insert(0, "date", dates)
        df.to_csv(output_dir / f"{name}.csv", index=False)
        saved_count += 1

    return saved_count


def _save_intermediate_process_outputs(
    details: dict,
    dates: pd.DatetimeIndex,
    subbasin_ids: list,
    output_dir: Path,
) -> None:
    """Save intermediate process outputs from model.forward(simulate=True)."""
    runoff_details = details.get("runoff", {}) if isinstance(details, dict) else {}
    states = runoff_details.get("states", {})
    fluxes = runoff_details.get("fluxes", {})

    process_dir = output_dir / "physical_processes"
    process_dir.mkdir(parents=True, exist_ok=True)

    states_dir = process_dir / "states"
    fluxes_dir = process_dir / "fluxes"
    states_dir.mkdir(exist_ok=True)
    fluxes_dir.mkdir(exist_ok=True)

    n_states = _save_tensor_series_group(states, dates, subbasin_ids, states_dir)
    n_fluxes = _save_tensor_series_group(fluxes, dates, subbasin_ids, fluxes_dir)

    manifest = {
        "n_timesteps": len(dates),
        "n_subbasins": len(subbasin_ids),
        "subbasin_ids": list(subbasin_ids),
        "state_variables_saved": n_states,
        "flux_variables_saved": n_fluxes,
        "state_variable_names": sorted([k for k, v in states.items() if isinstance(v, torch.Tensor)]),
        "flux_variable_names": sorted([k for k, v in fluxes.items() if isinstance(v, torch.Tensor)]),
    }
    with open(process_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Save] Intermediate process directory: {process_dir}")
    print(f"[Save] Intermediate files: states={n_states}, fluxes={n_fluxes}")


def simulate(
    config: dict,
    param_file: Optional[str] = None,
    basin_name: Optional[str] = None,
):
    """Run base simulation and save hydrological + optional process series."""
    _ = param_file  # kept for compatibility
    sim_config = config["simulation"]

    enabled_basins = get_enabled_basins(config)
    if not enabled_basins:
        raise ValueError("No enabled basin found in config.")

    if basin_name is None:
        basin_cfg = enabled_basins[0]
        basin_name = basin_cfg["name"]
    else:
        basin_cfg = next((b for b in enabled_basins if b["name"] == basin_name), enabled_basins[0])

    print(f"[Simulate] Basin: {basin_name}")
    data_paths = get_basin_data_paths(basin_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and sim_config.get("use_gpu", False) else "cpu"
    )
    print(f"[Simulate] Device: {device}")

    resolved_modes = resolve_component_param_modes(config)
    training_output_dir = Path(config["training"]["output_dir"])
    runtime_config = _build_simulation_runtime_config(
        config=config,
        resolved_modes=resolved_modes,
        basin_name=basin_name,
        training_output_dir=training_output_dir,
    )

    print("[Simulate] Param mode plan:")
    for component_name, resolved in resolved_modes.items():
        params_dir = runtime_config.get("_params_dir_override", "")
        extra = f", params_dir={params_dir}" if resolved.category == "traditional" and params_dir else ""
        print(f"  - {component_name}: {resolved.category}/{resolved.strategy}{extra}")

    model_config = copy.deepcopy(runtime_config)
    model_config["_model_attr_path"] = data_paths["model_attr"]
    model = HydroModel(data_paths["model_attr"], config=model_config)
    print(f"[Simulate] HydroModel initialized with {model.n_subbasins} subbasins")

    dl_model_configs = _build_dl_model_configs(config, resolved_modes)
    params_dict = None

    if dl_model_configs:
        network_dir = Path(sim_config.get("network_dir") or config["training"]["output_dir"])
        print(f"[Simulate] Loading trained networks from: {network_dir}")

        predictor = DynamicParameterPredictor(
            model_configs=dl_model_configs,
            data_dir=data_paths["dl_input"],
            enable_cold_region=config.get("model", {}).get("cold_region", {}).get("enabled", False),
            enable_precip_correct=config.get("model", {}).get("precip_correct", {}).get("enabled", False),
            verbose=True,
        )

        strict_network_loading = not bool(sim_config.get("allow_untrained_networks", False))
        _load_trained_networks(predictor, network_dir, device, strict=strict_network_loading)

        features = load_features(data_paths["dl_input"])
        with torch.no_grad():
            predicted_params = predictor.predict_params(features, training=False)
        print(f"[Simulate] Predicted parameters: {len(predicted_params)}")

        params_dict = _build_params_dict_for_sim(model, predicted_params)
    else:
        print("[Simulate] No DL components active; using fixed calibrated/manual parameters.")

    start_date, end_date = _resolve_simulation_period(sim_config, basin_cfg)
    print(f"[Simulate] Period: {start_date} to {end_date}")
    p, e, dates = load_forcing_data(basin_name, runtime_config, start_date=start_date, end_date=end_date)

    forcing = {"P": p, "E": e}
    temp_file = data_paths["forcing"]["temperature"]
    if os.path.exists(temp_file):
        t_df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
        t_df = t_df.loc[dates[0] : dates[-1]]
        forcing["T"] = t_df.values

    save_processes = bool(sim_config.get("save_physical_processes", True))

    print("[Simulate] Running forward simulation...")
    with torch.no_grad():
        outputs, _ = model.forward(
            forcing,
            params=params_dict,
            device=device,
            simulate=save_processes,
        )

    reach_flow = {k: v.cpu().numpy() for k, v in outputs["river_flows"].items()}
    subbasin_runoff = {k: v.cpu().numpy() for k, v in outputs["subbasin_runoff"].items()}
    print(f"[Simulate] Completed with {len(dates)} timesteps")

    output_dir = Path(sim_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    reach_series = {"date": dates}
    for reach_id, values in sorted(reach_flow.items()):
        reach_series[reach_id] = values
    pd.DataFrame(reach_series).to_csv(output_dir / "reach_flow_series.csv", index=False)

    subbasin_series = {"date": dates}
    for sub_id, values in sorted(subbasin_runoff.items()):
        subbasin_series[sub_id] = values
    pd.DataFrame(subbasin_series).to_csv(output_dir / "subbasin_runoff_series.csv", index=False)

    if save_processes:
        _save_intermediate_process_outputs(
            details=outputs.get("details", {}),
            dates=dates,
            subbasin_ids=model.subbasin_ids,
            output_dir=output_dir,
        )

    print(f"[Save] Output directory: {output_dir}")
    if save_processes:
        print("[Save] Files: reach_flow_series.csv, subbasin_runoff_series.csv, physical_processes/*")
    else:
        print("[Save] Files: reach_flow_series.csv, subbasin_runoff_series.csv")

    # Compatibility with upper-level caller signatures.
    return reach_flow, None, []


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Base hydrological simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--params", type=str, default=None, help="Reserved for compatibility")
    parser.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.start_date:
        config.setdefault("simulation", {})["start_date"] = args.start_date
    if args.end_date:
        config.setdefault("simulation", {})["end_date"] = args.end_date
    if args.output_dir:
        config.setdefault("simulation", {})["output_dir"] = args.output_dir

    simulate(config, args.params)
    print("[Done] Base simulation completed.")


if __name__ == "__main__":
    main()
