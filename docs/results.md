# Recorded results

## Ten-run sweep

The winning configuration was:

| Field | Value |
|---|---:|
| Learning rate | 0.0001 |
| Scheduler | cosine |
| Weight decay | 0.01 |
| Warmup ratio | 0.05 |
| Epochs | 10 |
| Validation loss | 3.445998 |
| Validation perplexity | 31.374571 |

## BabyLM Strict evaluation

| Evaluation | Metric | Value |
|---|---|---:|
| BLiMP | accuracy | 58.23% |
| BLiMP Supplement | accuracy | 54.68% |
| EWoK | accuracy | 51.19% |
| Entity Tracking | accuracy | 17.47% |
| COMPS | accuracy | 50.19% |
| GlobalPIQA parallel | accuracy | 28.16% |
| GlobalPIQA nonparallel | accuracy | 47.00% |
| BoolQ | accuracy | 69.30% |
| MNLI | accuracy | 46.98% |
| MRPC | F1 | 82.49% |
| MultiRC | accuracy | 61.39% |
| QQP | F1 | 62.20% |
| RTE | accuracy | 54.68% |
| WSC | accuracy | 61.54% |

These are the outputs of the official pipeline. Reading scores require care
when comparing models with different tokenizers; see the project report for
the separate word-boundary surprisal diagnostic.
