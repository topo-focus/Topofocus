from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch


@dataclass
class WrapperConfig:
    config_file: str
    dataset: str = "irsr"
    max_images: int = 800


class SalienceSingleImageWrapper:
    """Anonymous-release scaffold for single-image saliency retrieval."""

    def __init__(self, config_file: str, resume: bool = False, opts: Optional[List[str]] = None) -> None:
        _ = resume, opts
        self.cfg = WrapperConfig(config_file=config_file)
        self._iterator = self._empty_iterator()
        self._processed = 0

    def _empty_iterator(self) -> Iterator[Dict[str, Any]]:
        return iter(())

    def get_next_saliency(self) -> Optional[torch.Tensor]:
        if self._processed >= self.cfg.max_images:
            return None
        try:
            sample = next(self._iterator)
        except StopIteration:
            return None
        self._processed += 1
        return self._inference_single(sample)

    def _inference_single(self, sample: Dict[str, Any]) -> torch.Tensor:
        _ = sample
        raise RuntimeError(
            "Anonymous release placeholder: QAGNet single-image inference internals are withheld during review."
        )
