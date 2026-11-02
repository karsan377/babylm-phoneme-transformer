"""Compare phoneme_tokenizer.phonemize_many() with IPA-BabyLM on a random sample of rows.

Reports the exact-match rate, phones the tokenizer can't encode, and the most common remaining
differences (ours -> dataset). Keep phoneme_tokenizer.py in the same directory.

Run from the strict/ directory:  python check_phonemizer_bulk.py [N_ROWS]
(Leave APPEND_UTT_BOUNDARY = False in phoneme_tokenizer.py while running this check.)
"""

import collections
import difflib
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

from phoneme_tokenizer import phonemize_many

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CKPT = "../checkpoints/checkpoint-100000"

vocab = set(AutoTokenizer.from_pretrained(CKPT).get_vocab())

ds = load_dataset("phonemetransformers/IPA-BabyLM", "strict_small")["train"]
ds = ds.shuffle(seed=0).select(range(N * 2))
rows = [(r["text"], r["phonemized_utterance"]) for r in ds if isinstance(r["text"], str) and r["text"].strip()]
rows = rows[:N]

ours = phonemize_many([t for t, _ in rows])


def ws(s):
    return re.sub(r"\s+", " ", s).strip()


matches = 0
oov = collections.Counter()
subs = collections.Counter()
shown = 0
for (text, expected), got in zip(rows, ours):
    got, expected = ws(got), ws(expected)
    g, e = got.split(), expected.split()
    for p in g:
        if p not in vocab:
            oov[p] += 1
    if got == expected:
        matches += 1
        continue
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, g, e, autojunk=False).get_opcodes():
        if op != "equal":
            subs[(" ".join(g[a0:a1]) or "<none>", " ".join(e[b0:b1]) or "<none>")] += 1
    if shown < 3:
        shown += 1
        print("EXAMPLE MISMATCH:", repr(text[:80]))
        print("   ours    :", got[:160])
        print("   dataset :", expected[:160])

print(f"\nrows compared: {len(rows)}   exact match: {matches} ({matches / len(rows):.1%})")
print("\nphones we produce that are NOT in the tokenizer vocab (would become UNK):")
for p, c in oov.most_common(20):
    print(f"   {p!r}: {c}")
print("\nmost common remaining differences (ours -> dataset):")
for (a, b), c in subs.most_common(30):
    print(f"   {a!r} -> {b!r}: {c}")
