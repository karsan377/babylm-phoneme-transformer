#!/usr/bin/env python3
"""Train a randomly initialized Llama-style BabyLM with HF Trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


class PackedTokenDataset(Dataset):
    """Expose a memory-mapped token array as fixed-length causal-LM blocks."""

    def __init__(
        self,
        token_path: Path,
        word_ends_path: Path,
        block_size: int,
        seed: int,
        shuffle_once: bool,
    ):
        self.tokens = np.load(token_path, mmap_mode="r")
        word_ends = np.load(word_ends_path, mmap_mode="r")
        if len(word_ends) != len(self.tokens):
            raise ValueError(f"Token/word-end length mismatch: {token_path}")
        self.block_size = block_size
        self.num_blocks = math.ceil(len(self.tokens) / block_size)
        self.order = np.arange(self.num_blocks)
        if shuffle_once:
            np.random.default_rng(seed).shuffle(self.order)
        starts = np.arange(0, len(word_ends), block_size)
        counts = np.add.reduceat(word_ends, starts).astype(np.int64)
        self.word_counts = counts[self.order]

    def __len__(self) -> int:
        return len(self.order)

    def pad_to_multiple(self, multiple: int) -> None:
        """Repeat a few leading blocks so every optimizer step is full."""
        remainder = len(self.order) % multiple
        if remainder == 0:
            return
        needed = multiple - remainder
        self.order = np.concatenate([self.order, self.order[:needed]])
        self.word_counts = np.concatenate(
            [self.word_counts, self.word_counts[:needed]]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        block_index = int(self.order[index])
        start = block_index * self.block_size
        stop = min(start + self.block_size, len(self.tokens))
        values = np.asarray(self.tokens[start:stop], dtype=np.int64).copy()
        valid_length = len(values)
        if valid_length < self.block_size:
            values = np.pad(values, (0, self.block_size - valid_length), constant_values=0)

        input_ids = torch.from_numpy(values)
        attention_mask = torch.zeros(self.block_size, dtype=torch.long)
        attention_mask[:valid_length] = 1
        labels = input_ids.clone()
        labels[valid_length:] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class WordTracker:
    """Map optimizer steps to exact cumulative corpus-word exposure."""

    def __init__(self, block_word_counts: np.ndarray, effective_batch_size: int):
        self.words_per_epoch = int(block_word_counts.sum())
        step_counts = []
        for start in range(0, len(block_word_counts), effective_batch_size):
            step_counts.append(
                int(block_word_counts[start : start + effective_batch_size].sum())
            )
        self.step_word_counts = np.asarray(step_counts, dtype=np.int64)
        self.cumulative_one_epoch = np.cumsum(self.step_word_counts)
        self.steps_per_epoch = len(self.step_word_counts)

    def words_at_step(self, global_step: int) -> int:
        if global_step <= 0:
            return 0
        completed_epochs, within_epoch = divmod(global_step, self.steps_per_epoch)
        total = completed_epochs * self.words_per_epoch
        if within_epoch:
            total += int(self.cumulative_one_epoch[within_epoch - 1])
        return total


class DeterministicTrainer(Trainer):
    """Use the dataset's one seeded shuffle in the same order every epoch."""

    def _get_train_sampler(self):
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            return None
        return SequentialSampler(self.train_dataset)


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir: Path, word_tracker: WordTracker, total_words: int):
        self.output_dir = output_dir
        self.word_tracker = word_tracker
        self.total_words = total_words

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        record = dict(logs)
        record["global_step"] = state.global_step
        record["words_seen"] = min(
            self.total_words,
            self.word_tracker.words_at_step(state.global_step),
        )
        path = self.output_dir / "metrics.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


class WordMilestoneCallback(TrainerCallback):
    """Save lightweight model-only checkpoints at BabyLM word milestones."""

    def __init__(
        self,
        output_dir: Path,
        word_tracker: WordTracker,
        tokenizer,
        total_words: int,
    ):
        self.output_dir = output_dir
        self.word_tracker = word_tracker
        self.tokenizer = tokenizer
        candidates = [i * 1_000_000 for i in range(1, 11)]
        candidates += [i * 10_000_000 for i in range(2, 11)]
        self.milestones = [value for value in candidates if value <= total_words]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        words_seen = self.word_tracker.words_at_step(state.global_step)
        for milestone in self.milestones:
            target = self.output_dir / f"chck_{milestone // 1_000_000}M"
            if milestone > words_seen or target.exists():
                continue
            temporary = self.output_dir / f".{target.name}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            model.save_pretrained(temporary, safe_serialization=True)
            self.tokenizer.save_pretrained(temporary)
            metadata = {
                "target_words": milestone,
                "actual_words_seen": words_seen,
                "global_step": state.global_step,
            }
            (temporary / "milestone_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )
            os.replace(temporary, target)
            print(f"Saved {target.name} at {words_seen:,} words", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--run-name", default="hf-model-b")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Checkpoint path, 'auto' for newest checkpoint, or 'none'.",
    )
    parser.add_argument("--disable-word-milestones", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    manifest_words_per_epoch = int(manifest["counts"]["train"]["words"])

    tokenizer = AutoTokenizer.from_pretrained(args.data_dir / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = PackedTokenDataset(
        args.data_dir / "train_ids.npy",
        args.data_dir / "train_word_ends.npy",
        args.block_size,
        args.seed,
        shuffle_once=True,
    )
    eval_dataset = PackedTokenDataset(
        args.data_dir / "validation_ids.npy",
        args.data_dir / "validation_word_ends.npy",
        args.block_size,
        args.seed,
        shuffle_once=False,
    )

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=10,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=args.block_size,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    model = LlamaForCausalLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = args.batch_size * args.gradient_accumulation
    train_dataset.pad_to_multiple(effective_batch)
    word_tracker = WordTracker(train_dataset.word_counts, effective_batch)
    optimizer_steps_per_epoch = word_tracker.steps_per_epoch
    planned_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(optimizer_steps_per_epoch * args.epochs)
    )
    total_words = word_tracker.words_at_step(planned_steps)
    if total_words > 100_000_000:
        raise ValueError(f"Training would exceed the Strict budget: {total_words:,} words")
    run_config = {
        "model": config.to_dict(),
        "parameters": parameter_count,
        "train_blocks": len(train_dataset),
        "validation_blocks": len(eval_dataset),
        "words_per_epoch": word_tracker.words_per_epoch,
        "corpus_words_per_epoch": manifest_words_per_epoch,
        "repeated_padding_words_per_epoch": (
            word_tracker.words_per_epoch - manifest_words_per_epoch
        ),
        "planned_words": total_words,
        "effective_batch_size": effective_batch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_optimizer_steps": planned_steps,
        "arguments": vars(args) | {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
        },
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config, indent=2), flush=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=False,
    )

    callbacks = [MetricsCallback(args.output_dir, word_tracker, total_words)]
    if not args.disable_word_milestones:
        callbacks.append(
            WordMilestoneCallback(args.output_dir, word_tracker, tokenizer, total_words)
        )

    if args.resume_from_checkpoint == "auto":
        resume_checkpoint = get_last_checkpoint(str(args.output_dir))
    elif args.resume_from_checkpoint.lower() == "none":
        resume_checkpoint = None
    else:
        resume_checkpoint = args.resume_from_checkpoint
    print(f"Resume checkpoint: {resume_checkpoint}", flush=True)

    trainer = DeterministicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        eval_metrics["eval_perplexity"] = math.exp(min(eval_metrics["eval_loss"], 20))

    trainer.save_model(args.output_dir / "final")
    tokenizer.save_pretrained(args.output_dir / "final")
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
    print(json.dumps(eval_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
