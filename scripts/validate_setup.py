#!/usr/bin/env python3
"""Validate the isolated environment with a real one-step HF Trainer run."""

from __future__ import annotations

import importlib.util
import argparse
import json
import tempfile
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


EXPECTED = {
    "transformers": "4.45.2",
    "accelerate": "0.34.2",
    "numpy": "1.26.4",
}


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = torch.tensor([1, 4 + index, 6, 2], dtype=torch.long)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
            "labels": values.clone(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-data-check", action="store_true")
    args = parser.parse_args()
    actual = {
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for package, expected in EXPECTED.items():
        if actual[package] != expected:
            raise RuntimeError(
                f"{package} version mismatch: {actual[package]} != {expected}"
            )
    if importlib.util.find_spec("deepspeed") is not None:
        raise RuntimeError("DeepSpeed leaked into the isolated Hugging Face environment")

    if not args.skip_data_check:
        manifest = json.loads((args.data_dir / "manifest.json").read_text())
        token_ids = np.load(args.data_dir / "train_ids.npy", mmap_mode="r")
        word_ends = np.load(args.data_dir / "train_word_ends.npy", mmap_mode="r")
        if len(token_ids) != len(word_ends):
            raise RuntimeError("Token and word-end arrays do not align")
        if int(word_ends.sum()) != int(manifest["counts"]["train"]["words"]):
            raise RuntimeError("Word count does not match the data manifest")

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
        )
    )
    with tempfile.TemporaryDirectory(prefix="hf-trainer-preflight-") as output_dir:
        arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            max_steps=1,
            report_to=[],
            disable_tqdm=True,
            save_strategy="no",
            evaluation_strategy="no",
        )
        result = Trainer(
            model=model,
            args=arguments,
            train_dataset=TinyDataset(),
        ).train()
    if result.global_step != 1 or not np.isfinite(result.training_loss):
        raise RuntimeError(f"Invalid Trainer result: {result}")

    print(json.dumps(actual, indent=2))
    print("HF_TRAINER_SETUP_VALIDATED")


if __name__ == "__main__":
    main()
