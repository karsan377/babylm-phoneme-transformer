"""End-to-end check on held-out rows (shuffle seed 1; the map was built from seed 0).

For each dataset row, the wrapper applied to the raw text must give the same token ids as the
original tokenizer applied to the dataset's own `phonemized_utterance`.

Run from the strict/ directory after installing the wrapper into the checkpoint:
    python verify_wrapper.py [N_ROWS]
"""

import sys

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CKPT = "../checkpoints/checkpoint-100000"

new = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)  # tests the auto_map wiring
plain = PreTrainedTokenizerFast.from_pretrained(CKPT)              # original behaviour
print("loaded:", type(new).__name__, "| vocab:", len(new), "| special:", new.special_tokens_map)

ds = load_dataset("phonemetransformers/IPA-BabyLM", "strict_small")["train"]
ds = ds.shuffle(seed=1).select(range(N * 2))
rows = [(r["text"], r["phonemized_utterance"]) for r in ds if isinstance(r["text"], str) and r["text"].strip()]
rows = rows[:N]

texts = [t for t, _ in rows]
phons = [p for _, p in rows]
a_ids = new(texts)["input_ids"]
b_ids = plain(phons)["input_ids"]

unk = new.unk_token_id
same = sum(a == b for a, b in zip(a_ids, b_ids))
n_unk = sum(i == unk for ids in a_ids for i in ids)
n_tok = sum(len(ids) for ids in a_ids)

print(f"\nrows: {len(rows)}   identical token ids: {same} ({same / len(rows):.1%})")
print(f"UNK tokens from the wrapper: {n_unk} of {n_tok} ({n_unk / n_tok:.4%})")

shown = 0
for text, a, b in zip(texts, a_ids, b_ids):
    if a != b and shown < 3:
        shown += 1
        print("\nMISMATCH:", repr(text[:80]))
        print("   wrapper :", new.convert_ids_to_tokens(a)[:40])
        print("   original:", new.convert_ids_to_tokens(b)[:40])
