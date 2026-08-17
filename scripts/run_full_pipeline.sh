#!/bin/bash

set -euo pipefail

REPO_ROOT=<PATH_TO_REPO_ROOT>

echo "[Stage 1/2] Training QAGNet salient-object branch"
bash "${REPO_ROOT}/scripts/train_qagnet.sh"

echo "[Stage 2/2] Fine-tuning saliency-aware LLaVA"
bash "${REPO_ROOT}/scripts/finetune_lora.sh"

echo "Anonymous release note: executable internals are withheld during review."
