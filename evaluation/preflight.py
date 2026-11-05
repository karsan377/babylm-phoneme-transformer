#!/usr/bin/env python3
"""Fail-fast compatibility and data preflight for the BabyLM Strict evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--strict-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.strict_dir))

    required_model_files = [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
    ]
    for name in required_model_files:
        if not (args.model / name).is_file():
            raise FileNotFoundError(args.model / name)

    model_hash = sha256(args.model / "model.safetensors")
    if args.source_model:
        source_hash = sha256(args.source_model / "model.safetensors")
        if source_hash != model_hash:
            raise RuntimeError("Evaluation export weights differ from sweep winner")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer(
        ["The child reads a book.", "A dog runs."],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    encoded.pop("offset_mapping")
    # Some causal tokenizers (including the official GPT-2 baseline) emit
    # token_type_ids even though the strict evaluator's wrappers consume only
    # input_ids and attention_mask.
    encoded.pop("token_type_ids", None)
    if set(encoded) != {"input_ids", "attention_mask"}:
        raise RuntimeError(f"Unsupported tokenizer fields: {sorted(encoded)}")
    causal = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    config_vocab_size = int(causal.config.vocab_size)
    if len(tokenizer) != config_vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != "
            f"{config_vocab_size}"
        )
    causal_output = causal(
        **encoded, labels=encoded["input_ids"], output_hidden_states=True
    )
    if not torch.isfinite(causal_output.loss):
        raise RuntimeError("Non-finite causal loss")
    if causal_output.logits.shape[-1] != config_vocab_size:
        raise RuntimeError("Unexpected causal vocabulary dimension")

    base = AutoModel.from_pretrained(args.model, local_files_only=True)
    hidden = base(**encoded).last_hidden_state
    hidden_size = int(
        getattr(causal.config, "hidden_size", getattr(causal.config, "n_embd", 0))
    )
    if hidden.shape[-1] != hidden_size or not torch.isfinite(hidden).all():
        raise RuntimeError("Invalid AutoModel hidden states")

    # Exercise the evaluator's actual classification wrapper and gradients.
    from evaluation_pipeline.finetune.classifier_model import (
        ModelForSequenceClassification,
    )

    classifier_args = Namespace(
        model_name_or_path=str(args.model), revision_name=None,
        enc_dec=False, three_d_triangular_causal_mask=False,
        classifier_layer_norm_eps=1e-5, classifier_dropout=0.1,
        num_labels=2, take_final=True,
    )
    classifier = ModelForSequenceClassification(classifier_args)
    logits = classifier(encoded["input_ids"], encoded["attention_mask"])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite classifier loss")

    data_root = args.strict_dir / "evaluation_data" / "full_eval"
    jsonl_files = sorted(data_root.rglob("*.jsonl"))
    records = 0
    for path in jsonl_files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{path}:{line_number}: {error}") from error
                records += 1
    if len(jsonl_files) != 107 or records != 267_130:
        raise RuntimeError(
            f"Unexpected evaluation data: {len(jsonl_files)} files, {records} records"
        )

    # Exercise the official zero-shot data loader/scorer on real BLiMP records.
    from evaluation_pipeline.sentence_zero_shot.compute_results import compute_results
    from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader

    source = data_root / "blimp_filtered" / "anaphor_gender_agreement.jsonl"
    with tempfile.TemporaryDirectory(prefix="babylm-blimp-smoke-") as temp:
        smoke_dir = Path(temp) / "blimp_smoke"
        smoke_dir.mkdir()
        with source.open() as reader, (smoke_dir / source.name).open("w") as writer:
            for _ in range(4):
                writer.write(next(reader))
        smoke_args = Namespace(
            backend="causal", model_path_or_name=str(args.model),
            revision_name=None, task="blimp", data_path=smoke_dir,
            batch_size=4, non_causal_batch_size=4,
            full_sentence_scores=False, images_path=None, image_split=None,
            image_template=None, save_predictions=False,
        )
        loader = get_dataloader(smoke_args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        causal = causal.to(device)
        causal.eval()
        results, predictions = compute_results(smoke_args, causal, loader, [1.0])
        if not results or not predictions:
            raise RuntimeError("Official BLiMP smoke test returned no results")

    # Cover the reading-time path, including the previous-item branch that failed
    # silently in the earlier Pico evaluation.
    from evaluation_pipeline.reading.evaluation_functions import get_p2

    reading_rows = pd.read_csv(
        data_root / "reading" / "reading_data.csv", dtype={"item": str}
    ).head(12)
    reading_rows["item"] = reading_rows["item"].fillna("None")
    for index, row in reading_rows.iterrows():
        probability, _ = get_p2(row["item"], row["word"], causal, tokenizer)
        if not 0 < probability <= 1:
            raise RuntimeError(f"Invalid reading probability at row {index}: {probability}")
        if isinstance(row["prev_item"], str):
            probability, _ = get_p2(
                row["prev_item"], row["prev_word"], causal, tokenizer
            )
            if not 0 < probability <= 1:
                raise RuntimeError(
                    f"Invalid previous-item probability at row {index}: {probability}"
                )

    report = {
        "status": "passed",
        "model": str(args.model),
        "model_sha256": model_hash,
        "parameters": sum(parameter.numel() for parameter in causal.parameters()),
        "model_type": causal.config.model_type,
        "vocab_size": len(tokenizer),
        "context_length": tokenizer.model_max_length,
        "hidden_size": hidden.shape[-1],
        "causal_loss": float(causal_output.loss),
        "classifier_loss": float(loss),
        "jsonl_files": len(jsonl_files),
        "jsonl_records": records,
        "official_blimp_smoke_examples": 4,
        "reading_smoke_rows": len(reading_rows),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
