#!/usr/bin/env python
"""
single_image_inference.py
=========================
Minimal single-image inference with a Janus-style multimodal LLM,
used as one of the baseline VLMs we compare against in the paper.

The model identifier is withheld for anonymous review; replace
`<vlm_checkpoint>` with the public HuggingFace repo or local path.
"""

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


MODEL_PATH = "<vlm_checkpoint>"
IMAGE_PATH = "<path/to/image.jpg>"
PROMPT     = "Describe the most salient objects in this image in order of importance."


def build_model(model_path):
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()

    return model, processor, tokenizer


def run(model_path, image_path, prompt, max_new_tokens=512):
    model, processor, tokenizer = build_model(model_path)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image_path],
        },
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

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    response = run(MODEL_PATH, IMAGE_PATH, PROMPT)
    print(response)
