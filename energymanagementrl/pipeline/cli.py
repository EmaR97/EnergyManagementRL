import argparse
import importlib
import sys

from energymanagementrl.pipeline.config import load_config

_STEP_MODULES = {
    "all": "all",
    "ingest": "01_ingest",
    "forecast": "02_forecast",
    "process": "03_process",
    "train": "04_train",
    "evaluate": "05_evaluate",
    "deploy": "06_deploy",
    "monitor": "07_monitor",
}

_STEP_LABELS = {
    "all": "Run full pipeline",
    "ingest": "Ingest raw data from FusionSolar API",
    "forecast": "Generate Open-Meteo and clear-sky forecasts",
    "process": "Merge plant history with forecasts, analyze gaps, fill",
    "train": "Train DQN model",
    "evaluate": "Evaluate trained models",
    "deploy": "Deploy best model",
    "monitor": "Analyze production logs",
}

_STEP_ORDER = ["ingest", "forecast", "process", "train", "evaluate", "deploy", "monitor"]


def _import_step(name):
    return importlib.import_module(f"energymanagementrl.pipeline.{_STEP_MODULES[name]}")


def _show_menu():
    print("\n=== EnergyManagementRL Pipeline ===")
    print(f"  {'0:':3s} {'all':12s} Run full pipeline")
    for i, name in enumerate(_STEP_ORDER, 1):
        print(f"  {f'{i}:':3s} {name:12s} {_STEP_LABELS[name]}")
    print()

    while True:
        try:
            choice = input("Select step [0-7] (or q to quit): ").strip()
            if choice.lower() in ("q", ""):
                print("Aborted.")
                sys.exit(0)
            idx = int(choice)
            if idx == 0:
                return "all"
            if 1 <= idx <= len(_STEP_ORDER):
                return _STEP_ORDER[idx - 1]
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Try again.")


def main():
    parser = argparse.ArgumentParser(
        prog="empipeline",
        description="EnergyManagementRL pipeline — data ingestion through deployment",
    )
    sub = parser.add_subparsers(dest="command")

    for name, help in _STEP_LABELS.items():
        sub.add_parser(name, help=help)

    args, _ = parser.parse_known_args()

    command = args.command if args.command else _show_menu()
    config = load_config()

    if command == "all":
        for i, name in enumerate(_STEP_ORDER, 1):
            step_label = _STEP_LABELS.get(name, name)
            print(f"=== Step {i}: {step_label} ===")
            _import_step(name).run(config)
        print("=== Pipeline complete ===")
    else:
        _import_step(command).run(config)


if __name__ == "__main__":
    main()
