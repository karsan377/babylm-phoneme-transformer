#!/usr/bin/env python3
"""Collect the principal BabyLM report values into one portable CSV."""

import argparse
import csv
import json
import re
from pathlib import Path


ZERO_SHOT = {
    "BLiMP": "zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt",
    "BLiMP Supplement": "zero_shot/causal/blimp/supplement_filtered/best_temperature_report.txt",
    "EWoK": "zero_shot/causal/ewok/ewok_filtered/best_temperature_report.txt",
    "Entity Tracking": "zero_shot/causal/entity_tracking/entity_tracking/best_temperature_report.txt",
    "COMPS": "zero_shot/causal/comps/comps/best_temperature_report.txt",
    "GlobalPIQA parallel": "zero_shot/causal/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt",
    "GlobalPIQA nonparallel": "zero_shot/causal/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt",
}


def average_accuracy(path: Path) -> float:
    matches = re.findall(r"AVERAGE ACCURACY\s*\n([0-9.]+)", path.read_text())
    if not matches:
        raise RuntimeError(f"No average accuracy in {path}")
    return float(matches[-1]) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task, relative in ZERO_SHOT.items():
        rows.append(("zero_shot", task, "accuracy", average_accuracy(args.result_root / relative)))
    for task in ("boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"):
        path = args.result_root / "finetune" / task / "results.txt"
        text = path.read_text().strip()
        try:
            metrics = json.loads(text)
        except json.JSONDecodeError:
            metrics = dict(re.findall(r"([A-Za-z_]+)\s*[:=]\s*([-+0-9.eE]+)", text))
        primary = "f1" if task in {"mrpc", "qqp"} else "accuracy"
        if primary not in metrics:
            raise RuntimeError(f"Missing {primary} in {path}")
        rows.append(("finetune", task, primary, float(metrics[primary])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("evaluation_type", "task", "primary_metric", "primary_value"))
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
