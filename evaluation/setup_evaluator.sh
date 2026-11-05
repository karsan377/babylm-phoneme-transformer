#!/usr/bin/env bash
set -euo pipefail

REPOSITORY=${BABYLM_EVAL_REPOSITORY:-https://github.com/babylm-org/babylm-eval.git}
COMMIT=${BABYLM_EVAL_COMMIT:-6f825c291e2c4c78ad33b1935fd64d45f52642dc}
TARGET=${1:-external/babylm-eval}

if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone "$REPOSITORY" "$TARGET"
fi

git -C "$TARGET" fetch origin "$COMMIT"
git -C "$TARGET" checkout --detach "$COMMIT"
echo "BabyLM evaluator pinned at $(git -C "$TARGET" rev-parse HEAD)"
