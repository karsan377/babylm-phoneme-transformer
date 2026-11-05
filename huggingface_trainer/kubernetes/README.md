# Optional Kubernetes execution

These manifests are templates, not credentials. Replace `YOUR_NAMESPACE`,
`YOUR_REGISTRY`, and PVC placeholders with resources from your own cluster.
The evaluation template expects the pinned evaluator and downloaded evaluation
data on `YOUR_EVALUATOR_PVC`.
Never commit kubeconfig files, service-account tokens, or Kubernetes Secrets.

The Python entry points are identical locally and on Kubernetes. Kubernetes
only supplies compute and persistent volumes; it does not change the model.
