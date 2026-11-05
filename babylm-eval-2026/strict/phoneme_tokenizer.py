"""phoneme_tokenizer.py

Wraps the trained phoneme tokenizer so the BabyLM eval pipeline can feed it ordinary English
text. Each string is phonemized with espeak (en-us), normalised to the phone inventory used in
IPA-BabyLM, then encoded by the saved tokenizer.

PHONE_MAP was derived by comparing espeak output with the dataset on 2,000 rows
(check_phonemizer_bulk.py). Re-run that script after any change; it prints what is left over.
"""

import re

from transformers import PreTrainedTokenizerFast

RETRACTED = "\u0320"  # combining minus sign below, as in the vocab's affricates

# espeak phone -> the phone(s) the dataset uses (space-separated if one phone becomes several)
PHONE_MAP = {
    "ɚ": "ə ɹ",
    "ɐ": "ʌ",
    "ɑː": "ɑ",
    "əl": "ə l",
    "ᵻ": "ɪ",
    "ɔː": "ɔ",
    "oː": "ɔ",
    "ɑːɹ": "ɑ ɹ",
    "ɔːɹ": "ɔ ɹ",
    "oːɹ": "ɔ ɹ",
    "ɛɹ": "ɛ ɹ",
    "ʊɹ": "ʊ ɹ",
    "ɪɹ": "ɪ ɹ",
    "aɪɚ": "aɪ ə ɹ",
    "aɪə": "aɪ ə",
    "dʒ": f"d{RETRACTED}ʒ",
    "tʃ": f"t{RETRACTED}ʃ",
    "ɾ": "t",
    "ʔ": "t",
    "n̩": "ə n",
}

# True if training utterances ended with UTT_BOUNDARY (join_utts=None). Decide after checking
# your experiment config and how the eval pipeline scores sentences.
APPEND_UTT_BOUNDARY = False

_BACKEND = None
_CACHE = {}


def _backend():
    global _BACKEND
    if _BACKEND is None:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        _BACKEND = (
            EspeakBackend("en-us", preserve_punctuation=False, with_stress=False),
            Separator(phone=" ", word=" WORD_BOUNDARY ", syllable=""),
        )
    return _BACKEND


def _convert(raw):
    out = []
    for tok in re.sub(r"[ˈˌ]", "", raw).split():
        out.extend(PHONE_MAP.get(tok, tok).split())
    if out and out[-1] != "WORD_BOUNDARY":
        out.append("WORD_BOUNDARY")
    if APPEND_UTT_BOUNDARY:
        out.append("UTT_BOUNDARY")
    return " ".join(out)


def phonemize_many(texts):
    """List of English strings -> list of phoneme strings in the dataset's format (cached)."""
    todo = [t for t in dict.fromkeys(texts) if t not in _CACHE]
    if todo:
        backend, sep = _backend()
        raw = backend.phonemize([t.strip() for t in todo], separator=sep, strip=True)
        for t, r in zip(todo, raw):
            _CACHE[t] = _convert(r)
    return [_CACHE[t] for t in texts]


class PhonemeTokenizerFast(PreTrainedTokenizerFast):
    """Phonemizes text, then tokenizes with the saved phoneme vocabulary."""

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        if not kwargs.get("is_split_into_words", False):
            flat, shapes = [], []
            for item in batch_text_or_text_pairs:
                parts = [item] if isinstance(item, str) else list(item)
                shapes.append((isinstance(item, str), len(parts)))
                flat.extend(parts)
            phon = iter(phonemize_many(flat))
            rebuilt = []
            for is_single, n in shapes:
                parts = [next(phon) for _ in range(n)]
                rebuilt.append(parts[0] if is_single else tuple(parts))
            batch_text_or_text_pairs = rebuilt
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)
