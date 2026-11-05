#!/usr/bin/env python3
"""Run ten restart-safe, full-budget Model B hyperparameter trials."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TRAIN_SCRIPT = Path(__file__).with_name("train.py")

# Each trial changes one major optimization choice relative to nearby trials.
TRIALS = [
    {"name": "run01-lr1e-4-cos-wd01-wu05", "lr": "1e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.05"},
    {"name": "run02-lr2e-4-cos-wd01-wu05", "lr": "2e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.05"},
    {"name": "run03-lr3e-4-cos-wd01-wu05", "lr": "3e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.05"},
    {"name": "run04-lr4e-4-cos-wd01-wu05", "lr": "4e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.05"},
    {"name": "run05-lr5e-4-cos-wd01-wu05", "lr": "5e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.05"},
    {"name": "run06-lr3e-4-lin-wd01-wu05", "lr": "3e-4", "scheduler": "linear", "wd": "0.01", "warmup": "0.05"},
    {"name": "run07-lr3e-4-cos-wd00-wu05", "lr": "3e-4", "scheduler": "cosine", "wd": "0.00", "warmup": "0.05"},
    {"name": "run08-lr3e-4-cos-wd05-wu05", "lr": "3e-4", "scheduler": "cosine", "wd": "0.05", "warmup": "0.05"},
    {"name": "run09-lr3e-4-cos-wd01-wu02", "lr": "3e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.02"},
    {"name": "run10-lr3e-4-cos-wd01-wu10", "lr": "3e-4", "scheduler": "cosine", "wd": "0.01", "warmup": "0.10"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def write_summary(sweep_root: Path) -> list[dict]:
    rows = []
    for trial in TRIALS:
        output_dir = sweep_root / trial["name"]
        metrics_path = output_dir / "eval_results.json"
        row = dict(trial)
        row["output_dir"] = str(output_dir)
        if metrics_path.exists() and (output_dir / "final" / "config.json").exists():
            metrics = read_metrics(metrics_path)
            row.update(
                status="complete",
                eval_loss=metrics.get("eval_loss"),
                eval_perplexity=metrics.get("eval_perplexity"),
            )
        else:
            row["status"] = "pending"
        rows.append(row)
    complete = [row for row in rows if row["status"] == "complete"]
    complete.sort(key=lambda row: row["eval_loss"])
    for rank, row in enumerate(complete, start=1):
        row["rank"] = rank
    payload = {"updated_at": utc_now(), "trials": rows, "ranking": complete}
    (sweep_root / "sweep_summary.json").write_text(json.dumps(payload, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    sweep_root = args.output_dir
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "sweep_config.json").write_text(
        json.dumps({"epochs_per_trial": 10, "seed": 42, "trials": TRIALS}, indent=2)
    )
    write_summary(sweep_root)

    for index, trial in enumerate(TRIALS, start=1):
        output_dir = sweep_root / trial["name"]
        if (output_dir / "eval_results.json").exists() and (
            output_dir / "final" / "config.json"
        ).exists():
            print(f"SWEEP_SKIP {index}/10 {trial['name']} already complete", flush=True)
            continue

        print(f"SWEEP_START {index}/10 {trial['name']} {utc_now()}", flush=True)
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--output-dir", str(output_dir),
            "--data-dir", str(args.data_dir),
            "--epochs", str(args.epochs),
            "--learning-rate", trial["lr"],
            "--weight-decay", trial["wd"],
            "--warmup-ratio", trial["warmup"],
            "--lr-scheduler", trial["scheduler"],
            "--batch-size", "8",
            "--gradient-accumulation", "4",
            "--eval-steps", "250",
            "--save-steps", "1000",
            "--logging-steps", "20",
            "--run-name", trial["name"],
            "--resume-from-checkpoint", "auto",
            "--disable-word-milestones",
        ]
        if args.max_steps > 0:
            command.extend(["--max-steps", str(args.max_steps)])
        subprocess.run(command, check=True)
        write_summary(sweep_root)
        print(f"SWEEP_COMPLETE {index}/10 {trial['name']} {utc_now()}", flush=True)

    rows = write_summary(sweep_root)
    if sum(row["status"] == "complete" for row in rows) != len(TRIALS):
        raise RuntimeError("Sweep process exited without ten completed trials")
    print(f"SWEEP_ALL_COMPLETE {utc_now()}", flush=True)


if __name__ == "__main__":
    main()
