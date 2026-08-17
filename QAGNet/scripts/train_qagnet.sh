#!/bin/bash

cd <PATH_TO_REPO_ROOT>/QAGNet/src

python train_qagnet.py \
    --num-gpus <PUT_YOUR_NUM_GPUS> \
    --config-file ../configs/qagnet_r50.yaml \
    SOLVER.IMS_PER_BATCH <PUT_YOUR_BATCH_SIZE> \
    SOLVER.BASE_LR <PUT_YOUR_LR> \
    SOLVER.MAX_ITER <PUT_YOUR_MAX_ITER> \
    OUTPUT_DIR <PATH_TO_OUTPUT_DIR>
