# phoneme-strict-small-gpt2-85M results (phonemized BabyLM 2024 eval data)

Pairwise accuracy, chance = 50%.

| Checkpoint | BLiMP (filtered) | BLiMP (supplement) | EWoK (overall) |
|---|---|---|---|
| 150k (best val loss) | 65.2% | 57.2% | 49.4% |
| 200k (final) | 64.8% | 57.1% | 49.6% |

Training: final train loss 1.14. Validation loss 1.57 at 150k, 1.71 at 200k.

Training curves: W&B project phoneme-strict-small, run phoneme-strict-small-gpt2-85M.
Full logs: this folder (ksankaran-phoneme-*.log).
Checkpoints: /ckpt/phoneme-strict-small/phoneme-strict-small-gpt2-85M/ on claim ksankaran-phoneme-ckpt.
