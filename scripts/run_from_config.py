#!/usr/bin/env python3
"""Translate a small YAML config into the canonical train.py CLI."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    script = Path(__file__).with_name("train.py")
    command = [sys.executable, str(script)]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key.replace("_", "-"))
        else:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
