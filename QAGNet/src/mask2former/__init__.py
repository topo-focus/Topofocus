from __future__ import annotations

from typing import Any


class MaskFormerInstanceDatasetMapper:
    def __init__(self, cfg: Any, is_train: bool = True) -> None:
        self.cfg = cfg
        self.is_train = is_train

    def __call__(self, item: Any) -> Any:
        return item


class COCOInstanceNewBaselineDatasetMapper(MaskFormerInstanceDatasetMapper):
    pass


class InstanceSegEvaluator:
    def __init__(self, dataset_name: str, output_dir: str = None) -> None:
        self.dataset_name = dataset_name
        self.output_dir = output_dir


def add_maskformer2_config(cfg: Any) -> None:
    # Placeholder hook to match the expected Detectron2 extension API.
    _ = cfg
