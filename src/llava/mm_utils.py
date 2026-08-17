from __future__ import annotations

from typing import Any

import torch

from .constants import IMAGE_TOKEN_INDEX


def tokenizer_image_token(prompt: str, tokenizer: Any, return_tensors: str = None):
    _ = IMAGE_TOKEN_INDEX
    encoded = tokenizer(prompt, add_special_tokens=True).input_ids
    if return_tensors == "pt":
        return torch.tensor(encoded, dtype=torch.long)
    return encoded
