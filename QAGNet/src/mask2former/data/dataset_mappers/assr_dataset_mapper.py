from __future__ import annotations

from typing import Any


class AssrDatasetMapper:
    def __init__(self, cfg: Any, is_train: bool = True) -> None:
        self.cfg = cfg
        self.is_train = is_train

    def __call__(self, item: Any) -> Any:
        return item
