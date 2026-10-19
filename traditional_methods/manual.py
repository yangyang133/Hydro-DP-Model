from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module
from utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Validate manual traditional calibration inputs")
    parser.add_argument("--config", type=str, required=True, help="Runtime YAML config path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = train_module.load_config(args.config)
    set_global_seed(int(config["training"]["seed"]))

    enabled_basins = train_module.get_enabled_basins(config)
    if not enabled_basins:
        raise ValueError("No enabled basins found in configuration!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_summaries = []
    for basin_cfg in enabled_basins:
        basin_data = train_module.load_basin_data(basin_cfg, config)
        basin_summaries.append(
            {
                "name": basin_data.name,
                "description": basin_data.description,
                "role": basin_data.role,
                "weight": basin_data.weight,
                "n_subbasins": int(basin_data.model.n_subbasins),
                "n_reaches": int(len(basin_data.model.reaches)),
            }
        )

    summary = {
        "mode": "traditional/manual",
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_basins": len(basin_summaries),
        "basins": basin_summaries,
        "message": "Manual mode validated fixed-parameter inputs; no optimization was run.",
    }

    summary_path = output_dir / "manual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Manual] Validated {len(basin_summaries)} basin(s)")
    print(f"[Manual] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
