# Architecture

`scripts/train.py` constructs `LlamaForCausalLM` from a `LlamaConfig`; no
pretrained weights are loaded. The model has 10 decoder layers, hidden size
512, MLP size 1,536, eight attention heads, four key/value heads, tied input
and output embeddings, rotary position embeddings, and a 512-token context.

The dataset is a memory-mapped token stream divided into fixed-length blocks.
A parallel Boolean array marks the final token of each corpus word. The
`WordTracker` converts optimizer steps into cumulative word exposure so the
training process can enforce BabyLM's Strict budget and save milestone models.

Hugging Face `Trainer` provides mixed precision, gradient accumulation,
evaluation, checkpointing, and optimizer/scheduler state restoration. A
deterministic sequential sampler preserves the single seeded block order.
