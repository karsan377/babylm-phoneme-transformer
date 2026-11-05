#!/usr/bin/env python3
"""Print and optionally export the lowest-validation-loss sweep model."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    runs = []
    for metrics_path in args.sweep_dir.glob("run*/eval_results.json"):
        metrics = json.loads(metrics_path.read_text())
        final = metrics_path.parent / "final"
        if final.joinpath("config.json").is_file():
            runs.append((float(metrics["eval_loss"]), metrics_path.parent, metrics))
    if not runs:
        raise RuntimeError(f"No completed sweep runs under {args.sweep_dir}")
    loss, run, metrics = min(runs, key=lambda item: item[0])
    print(json.dumps({"best_run": str(run), **metrics}, indent=2))
    if args.export_dir:
        if args.export_dir.exists():
            raise FileExistsError(args.export_dir)
        shutil.copytree(run / "final", args.export_dir)


if __name__ == "__main__":
    main()
