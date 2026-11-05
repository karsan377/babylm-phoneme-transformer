#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${1:?Usage: run_full_eval.sh MODEL_DIR [STRICT_DIR]}
STRICT_DIR=${2:-external/babylm-eval/strict}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
STRICT_DIR=$(cd "$STRICT_DIR" && pwd)
MODEL_NAME=$(basename "$MODEL_DIR")
RESULT_ROOT="$STRICT_DIR/results/$MODEL_NAME/main"

cd "$STRICT_DIR"
python "$ROOT/evaluation/preflight.py" --model "$MODEL_DIR" --strict-dir "$STRICT_DIR"
bash scripts/eval_zero_shot.sh "$MODEL_DIR" causal evaluation_data/full_eval
test -s "$RESULT_ROOT/zero_shot/causal/reading/report.txt"

for task in global_piqa_parallel global_piqa_nonparallel; do
  python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_DIR" --backend causal --task "$task" \
    --data_path "evaluation_data/full_eval/$task" --save_predictions
done

bash scripts/eval_finetuning.sh --model_path "$MODEL_DIR"
for task in boolq multirc rte wsc mrpc qqp mnli; do
  test -s "$RESULT_ROOT/finetune/$task/results.txt"
  test -s "$RESULT_ROOT/finetune/$task/predictions.json"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$RESULT_ROOT/EVALUATION_COMPLETE"
echo "FULL_EVALUATION_COMPLETE: $RESULT_ROOT"
