from pathlib import Path

import numpy as np

from scripts.prepare_data import encode_split


class CharacterTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": list(range(10, 10 + len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_word_count_alignment(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_text("two words\nthree small words\n")
    counts = encode_split(CharacterTokenizer(), [source], tmp_path / "train")
    ids = np.load(tmp_path / "train_ids.npy")
    ends = np.load(tmp_path / "train_word_ends.npy")
    assert len(ids) == len(ends)
    assert counts["words"] == int(ends.sum()) == 5
