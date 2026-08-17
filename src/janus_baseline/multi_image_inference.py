#!/usr/bin/env python
"""
multi_image_inference.py
========================
Batched / multi-image inference with a Janus-style VLM used as a
baseline in the paper. Loads N images per sample, builds a single
multimodal turn referencing all of them, and decodes a response.

The model identifier and any task-specific prompt content are
withheld for anonymous review.
"""

import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images


MODEL_PATH      = "<vlm_checkpoint>"
INPUT_FOLDER    = "<path/to/sample/folder>"
NUM_IMAGES      = 4
MAX_NEW_TOKENS  = 512


# ---------------------------------------------------------------------------
# Model loading (eager attention to avoid Flash-Attn kernel issues on
# older GPUs - drop this if your build supports it)
# ---------------------------------------------------------------------------
def initialize_model(model_path=MODEL_PATH):
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "language_config"):
        config.language_config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    return model, processor, tokenizer


# ---------------------------------------------------------------------------
# Frame discovery - tries a few common naming conventions
# ---------------------------------------------------------------------------
def load_first_n_frames(folder_path: str, n: int) -> List[str]:
    folder = Path(folder_path)
    paths = []
    for i in range(n):
        candidates = [
            f"{i}.jpg",
            f"{i:02d}.jpg",
            f"{i:04d}.jpg",
        ]
        for name in candidates:
            p = folder / name
            if p.exists():
                paths.append(str(p))
                break
        else:
            raise FileNotFoundError(
                f"frame {i} not found in {folder_path}; tried {candidates}"
            )
    return paths


# ---------------------------------------------------------------------------
# Multi-image inference
# ---------------------------------------------------------------------------
def run_multi_image(
    image_paths: List[str],
    model,
    processor,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    placeholders = "\n".join(["<image_placeholder>"] * len(image_paths))
    user_content = f"{placeholders}\n{prompt}"

    conversation = [
        {"role": "User", "content": user_content, "images": image_paths},
        {"role": "Assistant", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    model, processor, tokenizer = initialize_model()

    image_paths = load_first_n_frames(INPUT_FOLDER, NUM_IMAGES)
    print(f"[INFO] using {len(image_paths)} images: "
          f"{[os.path.basename(p) for p in image_paths]}")

    prompt = "<task-specific prompt withheld for review>"
    response = run_multi_image(
        image_paths, model, processor, tokenizer, prompt
    )

    print(response)


if __name__ == "__main__":
    main()
