#!/bin/bash

REPO_ROOT=<PATH_TO_REPO_ROOT>

python "${REPO_ROOT}/QAGNet/src/inference_salience.py" \
  --config-file "${REPO_ROOT}/QAGNet/configs/qagnet_eval.yaml" \
  --model-root "<PATH_TO_QAGNET_WEIGHTS>" \
  --dataset-mode "<PUT_YOUR_SPLIT>"
