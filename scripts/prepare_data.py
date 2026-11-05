#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer and create packed token/word arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

WORD = re.compile(r"\S+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_lines(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield text


def build_tokenizer(train_files: list[Path], output: Path, vocab_size: int):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
    )
    tokenizer.train_from_iterator(iter_lines(train_files), trainer=trainer)
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", mask_token="<mask>", model_max_length=512,
    )
    wrapped.save_pretrained(output)
    return wrapped


def encode_split(tokenizer, files: list[Path], prefix: Path) -> dict:
    ids: list[int] = []
    word_ends: list[int] = []
    words = 0
    lines = 0
    for text in iter_lines(files):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = encoded["input_ids"] + [tokenizer.eos_token_id]
        offsets = encoded["offset_mapping"]
        ends = {match.end() for match in WORD.finditer(text)}
        flags = [int(end in ends) for _, end in offsets] + [0]
        ids.extend(token_ids)
        word_ends.extend(flags)
        words += len(ends)
        lines += 1
    np.save(prefix.with_name(prefix.name + "_ids.npy"), np.asarray(ids, dtype=np.int32))
    np.save(prefix.with_name(prefix.name + "_word_ends.npy"), np.asarray(word_ends, dtype=np.uint8))
    return {"files": len(files), "lines": lines, "tokens": len(ids), "words": words}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    for path in args.train + args.validation:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer = build_tokenizer(args.train, tokenizer_dir, args.vocab_size)
    counts = {
        "train": encode_split(tokenizer, args.train, args.output_dir / "train"),
        "validation": encode_split(tokenizer, args.validation, args.output_dir / "validation"),
    }
    manifest = {
        "vocab_size": len(tokenizer),
        "counts": counts,
        "sources": {
            "train": [{"path": str(p), "sha256": sha256(p)} for p in args.train],
            "validation": [{"path": str(p), "sha256": sha256(p)} for p in args.validation],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
