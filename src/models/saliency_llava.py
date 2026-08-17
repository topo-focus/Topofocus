from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class _StubImageProcessor:
    image_mean = (0.5, 0.5, 0.5)
    crop_size = {"height": 224, "width": 224}

    def preprocess(self, image: Any, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        _ = image, return_tensors
        return {"pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float32)}


class _StubVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_processor = _StubImageProcessor()


class SaliencyLlavaForCausalLM(nn.Module):
    """Anonymous-release scaffold preserving the public training interface."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = nn.Module()
        self.model.mm_projector = nn.Linear(32, 32, bias=False)
        self.sal_projector = nn.Linear(32, 32, bias=False)
        self._input_embeddings = nn.Embedding(32000, 32)
        self._output_head = nn.Linear(32, 32000, bias=False)
        self._vision_tower = _StubVisionTower()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args: Any, **kwargs: Any) -> "SaliencyLlavaForCausalLM":
        _ = model_name_or_path, args, kwargs
        return cls()

    def get_model(self) -> nn.Module:
        return self.model

    def get_vision_tower(self) -> _StubVisionTower:
        return self._vision_tower

    def initialize_vision_modules(self, model_args: Optional[Any] = None, fsdp: Optional[Any] = None) -> None:
        _ = model_args, fsdp

    def initialize_vision_tokenizer(self, model_args: Any, tokenizer: Any) -> None:
        _ = model_args, tokenizer

    def get_input_embeddings(self) -> nn.Embedding:
        return self._input_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self._output_head

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        old_dim = self._input_embeddings.embedding_dim
        self._input_embeddings = nn.Embedding(new_num_tokens, old_dim)
        self._output_head = nn.Linear(old_dim, new_num_tokens, bias=False)

    def save_pretrained(self, output_dir: str, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        _ = output_dir, state_dict
        # Intentionally a no-op in anonymous release.

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        _ = args, kwargs
        raise RuntimeError(
            "Anonymous release placeholder: core Saliency-LLaVA forward logic is withheld during review."
        )
