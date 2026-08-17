#!/bin/bash

REPO_ROOT=<PATH_TO_REPO_ROOT>

deepspeed ${REPO_ROOT}/src/train_saliency_llava.py \
    --lora_enable True --lora_r <PUT_YOUR_LORA_R> --lora_alpha <PUT_YOUR_LORA_ALPHA> --mm_projector_lr <PUT_YOUR_MM_PROJECTOR_LR> \
    --deepspeed ${REPO_ROOT}/configs/zero3.json \
    --model_name_or_path <PATH_TO_BASE_LLAVA_CKPT> \
    --version v1 \
    --data_path <PATH_TO_TRAIN_JSON> \
    --image_folder <PATH_TO_IMAGE_FOLDER> \
    --saliency_mask_folder <PATH_TO_SALIENCY_MASK_FOLDER> \
    --vision_tower <VISION_TOWER_NAME> \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir <PATH_TO_OUTPUT_DIR> \
    --num_train_epochs <PUT_YOUR_EPOCH_NUMBER> \
    --per_device_train_batch_size <PUT_YOUR_BATCH_SIZE> \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps <PUT_YOUR_GRAD_ACCUM_STEPS> \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps <PUT_YOUR_SAVE_STEPS> \
    --save_total_limit 1 \
    --learning_rate <PUT_YOUR_LR> \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length <PUT_YOUR_MAX_LENGTH> \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
