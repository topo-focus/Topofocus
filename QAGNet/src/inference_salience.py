from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List

import torch


def find_all_indexes(items: List[Any], value: Any) -> List[int]:
    return [idx for idx, item in enumerate(items) if item == value]


@contextmanager
def inference_context(model: Any):
    training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        yield
    finally:
        if hasattr(model, "train"):
            model.train(training)


def trivial_batch_collator(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def inference(cfg: Any, model: Any, model_name: str, model_root_dir: str, datasetmode: str):
    _ = cfg, model, model_name, model_root_dir, datasetmode
    raise RuntimeError(
        "Anonymous release placeholder: saliency-map inference and evaluation path is intentionally omitted."
    )
