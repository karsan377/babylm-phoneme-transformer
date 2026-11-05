# Reproduction levels

## Level 1: smoke reproduction

Run preprocessing and five training steps on tiny text files. This verifies
tokenization, word accounting, Trainer integration, checkpoint creation,
evaluation, and model export on a laptop.

## Level 2: full local reproduction

Prepare the official Strict corpus, run the ten configurations, select the
lowest validation-loss model, and run the official evaluator. A modern GPU
and substantial runtime are required, but no Kubernetes access is required.

## Level 3: optional cluster reproduction

Build the Docker image and fill in the generic Kubernetes placeholders using
the reproducer's own namespace, registry, and PVCs. Kubernetes is only a
compute backend; it is not part of the model definition.

## Determinism and recorded state

The scripts record the seed, configuration, model architecture, exact planned
word exposure, metrics, Trainer state, and source file hashes. Exact floating
point values can still vary slightly across GPU architectures and CUDA/PyTorch
builds. Model selection must use validation loss, never the official test
scores.
